#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/imgcodecs.hpp>

#include <linux/videodev2.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

class Gemini2MjpegNode : public rclcpp::Node
{
public:
  Gemini2MjpegNode() : Node("gemini2_mjpeg")
  {
    device_ = declare_parameter<std::string>("device", "auto");
    width_ = declare_parameter<int>("width", 1280);
    height_ = declare_parameter<int>("height", 720);
    fps_ = declare_parameter<int>("fps", 60);
    raw_preview_fps_ = declare_parameter<int>("raw_preview_fps", 10);
    frame_id_ = declare_parameter<std::string>("frame_id", "gemini2_color_optical_frame");
    compressed_topic_ = declare_parameter<std::string>(
      "topic", "/gemini2/color/image_raw/compressed");
    raw_topic_ = declare_parameter<std::string>("raw_topic", "/gemini2/color/image_raw");
    compressed_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
      compressed_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).reliable());
    raw_pub_ = create_publisher<sensor_msgs::msg::Image>(
      raw_topic_, rclcpp::QoS(rclcpp::KeepLast(2)).reliable());
    open_camera();
    configure_camera();
    init_mmap();
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(VIDIOC_STREAMON, &type);
    running_ = true;
    worker_ = std::thread(&Gemini2MjpegNode::capture_loop, this);
    RCLCPP_INFO(
      get_logger(), "Publishing Gemini 2 MJPEG: %dx%d @ %d FPS from %s",
      width_, height_, fps_, device_.c_str());
  }

  ~Gemini2MjpegNode() override
  {
    running_ = false;
    if (worker_.joinable()) worker_.join();
    if (fd_ >= 0) {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      ioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    for (auto & buffer : buffers_) {
      if (buffer.start && buffer.length) munmap(buffer.start, buffer.length);
    }
    if (fd_ >= 0) close(fd_);
  }

private:
  struct Buffer {void * start = nullptr; size_t length = 0;};

  static int retry_ioctl(int fd, unsigned long request, void * arg)
  {
    int result;
    do {result = ioctl(fd, request, arg);} while (result == -1 && errno == EINTR);
    return result;
  }

  void xioctl(unsigned long request, void * arg)
  {
    if (retry_ioctl(fd_, request, arg) == -1) {
      throw std::runtime_error(std::string("V4L2 ioctl failed: ") + std::strerror(errno));
    }
  }

  static uint32_t capabilities(const v4l2_capability & cap)
  {
    return (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
  }

  static bool usable_capture(const v4l2_capability & cap)
  {
    const auto bits = capabilities(cap);
    return (bits & V4L2_CAP_VIDEO_CAPTURE) && (bits & V4L2_CAP_STREAMING);
  }

  static bool is_gemini(const v4l2_capability & cap)
  {
    std::string card(reinterpret_cast<const char *>(cap.card));
    std::transform(card.begin(), card.end(), card.begin(),
      [](unsigned char c) {return static_cast<char>(std::tolower(c));});
    // Kernel/UVC versions use different card strings (e.g. "Orbbec RGB
    // Camera" or "Gemini 2").  Requiring both names incorrectly rejects the
    // valid RGB node, so either vendor/product marker is sufficient here.
    return card.find("orbbec") != std::string::npos ||
           card.find("gemini") != std::string::npos;
  }

  static std::vector<std::string> candidate_devices()
  {
    std::vector<std::string> candidates;
    std::error_code error;
    for (const char * directory : {"/dev/v4l/by-id", "/dev/v4l/by-path"}) {
      std::filesystem::directory_iterator entries(directory, error);
      if (error) {
        error.clear();
        continue;
      }
      for (const auto & entry : entries) {
        const std::string name = entry.path().filename().string();
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
          [](unsigned char c) {return static_cast<char>(std::tolower(c));});
        if (lower.find("orbbec") != std::string::npos ||
            lower.find("gemini") != std::string::npos) {
          candidates.push_back(entry.path().string());
        }
      }
    }
    for (int index = 0; index < 64; ++index) {
      candidates.push_back("/dev/video" + std::to_string(index));
    }
    return candidates;
  }

  static bool has_mode(int fd, int width, int height, int fps)
  {
    for (uint32_t fi = 0; ; ++fi) {
      v4l2_fmtdesc format{};
      format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      format.index = fi;
      if (retry_ioctl(fd, VIDIOC_ENUM_FMT, &format) < 0) break;
      if (format.pixelformat != V4L2_PIX_FMT_MJPEG) continue;
      for (uint32_t si = 0; ; ++si) {
        v4l2_frmsizeenum size{};
        size.pixel_format = V4L2_PIX_FMT_MJPEG;
        size.index = si;
        if (retry_ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) < 0) break;
        if (size.type != V4L2_FRMSIZE_TYPE_DISCRETE ||
          size.discrete.width != static_cast<uint32_t>(width) ||
          size.discrete.height != static_cast<uint32_t>(height)) continue;
        for (uint32_t ii = 0; ; ++ii) {
          v4l2_frmivalenum interval{};
          interval.pixel_format = V4L2_PIX_FMT_MJPEG;
          interval.width = width;
          interval.height = height;
          interval.index = ii;
          if (retry_ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &interval) < 0) break;
          if (interval.type == V4L2_FRMIVAL_TYPE_DISCRETE &&
            interval.discrete.numerator != 0) {
            const double found = static_cast<double>(interval.discrete.denominator) /
              interval.discrete.numerator;
            if (std::abs(found - fps) < 0.01) return true;
          }
        }
      }
    }
    return false;
  }

  void open_camera()
  {
    if (device_ == "auto") {
      for (const auto & path : candidate_devices()) {
        const int candidate = open(path.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (candidate < 0) continue;
        v4l2_capability cap{};
        const bool usable = retry_ioctl(candidate, VIDIOC_QUERYCAP, &cap) == 0 &&
          usable_capture(cap) && is_gemini(cap) && has_mode(candidate, width_, height_, fps_);
        if (usable) {
          fd_ = candidate;
          device_ = path;
          break;
        }
        close(candidate);
      }
    } else {
      fd_ = open(device_.c_str(), O_RDWR | O_NONBLOCK, 0);
    }
    if (fd_ < 0) {
      std::ostringstream message;
      message << "No Gemini 2 RGB V4L2 node provides requested "
              << width_ << "x" << height_ << " MJPEG @ " << fps_ << " FPS";
      throw std::runtime_error(message.str());
    }
    v4l2_capability cap{};
    xioctl(VIDIOC_QUERYCAP, &cap);
    if (!usable_capture(cap)) throw std::runtime_error(device_ + " is not a streaming camera");
    RCLCPP_INFO(get_logger(), "Auto-selected Gemini RGB device: %s", device_.c_str());
  }

  void configure_camera()
  {
    const int wanted_width = width_;
    const int wanted_height = height_;
    const int wanted_fps = fps_;
    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = wanted_width;
    format.fmt.pix.height = wanted_height;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    xioctl(VIDIOC_S_FMT, &format);
    width_ = static_cast<int>(format.fmt.pix.width);
    height_ = static_cast<int>(format.fmt.pix.height);
    if (format.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG ||
      width_ != wanted_width || height_ != wanted_height) {
      throw std::runtime_error("Gemini refused exact MJPEG resolution");
    }
    v4l2_streamparm request{};
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.parm.capture.timeperframe.numerator = 1;
    request.parm.capture.timeperframe.denominator = wanted_fps;
    xioctl(VIDIOC_S_PARM, &request);
    v4l2_streamparm actual{};
    actual.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(VIDIOC_G_PARM, &actual);
    const double real_fps = actual.parm.capture.timeperframe.numerator == 0 ? 0.0 :
      static_cast<double>(actual.parm.capture.timeperframe.denominator) /
      actual.parm.capture.timeperframe.numerator;
    if (std::abs(real_fps - wanted_fps) >= 0.01) {
      std::ostringstream text;
      text << "Gemini refused " << wanted_fps << " FPS; got " << real_fps;
      throw std::runtime_error(text.str());
    }
    RCLCPP_INFO(
      get_logger(), "V4L2 actual format: %dx%d MJPEG @ %.3f FPS", width_, height_, real_fps);
  }

  void init_mmap()
  {
    v4l2_requestbuffers request{};
    request.count = 6;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    xioctl(VIDIOC_REQBUFS, &request);
    if (request.count < 2) throw std::runtime_error("Gemini returned too few V4L2 buffers");
    buffers_.resize(request.count);
    for (uint32_t index = 0; index < request.count; ++index) {
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = index;
      xioctl(VIDIOC_QUERYBUF, &buffer);
      buffers_[index].length = buffer.length;
      buffers_[index].start = mmap(
        nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
      if (buffers_[index].start == MAP_FAILED) throw std::runtime_error("Gemini mmap failed");
      xioctl(VIDIOC_QBUF, &buffer);
    }
  }

  void publish_frame(const uint8_t * bytes, size_t jpeg_size)
  {
    const auto stamp = now();
    sensor_msgs::msg::CompressedImage compressed;
    compressed.header.stamp = stamp;
    compressed.header.frame_id = frame_id_;
    compressed.format = "bgr8; jpeg compressed bgr8";
    compressed.data.assign(bytes, bytes + jpeg_size);
    compressed_pub_->publish(std::move(compressed));

    // Raw is only a low-rate operator/RViz preview.
    // FTP-1 recording consumes the 60 Hz CompressedImage directly.
    if (raw_pub_->get_subscription_count() + raw_pub_->get_intra_process_subscription_count() == 0) return;
    if (raw_preview_fps_ <= 0) return;

    const auto raw_now = std::chrono::steady_clock::now();
    const auto raw_period = std::chrono::duration<double>(
      1.0 / static_cast<double>(raw_preview_fps_));

    if (last_raw_publish_.time_since_epoch().count() != 0 &&
      raw_now - last_raw_publish_ < raw_period) {
      return;
    }
    last_raw_publish_ = raw_now;

    const cv::Mat encoded(1, static_cast<int>(jpeg_size), CV_8UC1, const_cast<uint8_t *>(bytes));
    cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (decoded.empty() || decoded.cols != width_ || decoded.rows != height_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Gemini MJPEG decode failed");
      return;
    }
    if (!decoded.isContinuous()) decoded = decoded.clone();
    sensor_msgs::msg::Image raw;
    raw.header.stamp = stamp;
    raw.header.frame_id = frame_id_;
    raw.height = static_cast<uint32_t>(decoded.rows);
    raw.width = static_cast<uint32_t>(decoded.cols);
    raw.encoding = "bgr8";
    raw.is_bigendian = false;
    raw.step = static_cast<uint32_t>(decoded.cols * decoded.elemSize());
    raw.data.assign(decoded.datastart, decoded.dataend);
    raw_pub_->publish(std::move(raw));
    ++raw_count_;
  }

  void capture_loop()
  {
    auto last_report = std::chrono::steady_clock::now();
    bool device_lost = false;
    while (running_ && rclcpp::ok()) {
      pollfd descriptor{fd_, POLLIN, 0};
      const int ready = poll(&descriptor, 1, 1000);
      if (ready <= 0) {
        if (ready == 0) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Gemini frame timeout");
        }
        else if (errno != EINTR) {
          const int error = errno;
          if (error == ENODEV || error == ENXIO || error == EBADF) {
            RCLCPP_ERROR(get_logger(), "Gemini device disconnected during poll: %s", std::strerror(error));
            device_lost = true;
            break;
          }
          RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 2000, "Gemini poll failed: %s", std::strerror(error));
        }
        continue;
      }
      if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        RCLCPP_ERROR(
          get_logger(), "Gemini device became unavailable (poll events: 0x%x)", descriptor.revents);
        device_lost = true;
        break;
      }
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      if (retry_ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
        const int error = errno;
        if (error == ENODEV || error == ENXIO || error == EBADF) {
          RCLCPP_ERROR(
            get_logger(), "Gemini device disconnected during dequeue: %s", std::strerror(error));
          device_lost = true;
          break;
        }
        if (error != EAGAIN) {
          RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 2000, "Gemini dequeue failed: %s", std::strerror(error));
        }
        continue;
      }
      if (!(buffer.flags & V4L2_BUF_FLAG_ERROR) && buffer.index < buffers_.size() &&
        buffer.bytesused > 4 && buffer.bytesused <= buffers_[buffer.index].length) {
        const auto * bytes = static_cast<const uint8_t *>(buffers_[buffer.index].start);
        size_t jpeg_size = 0;
        if (bytes[0] == 0xff && bytes[1] == 0xd8) {
          for (size_t i = buffer.bytesused - 1; i > 0; --i) {
            if (bytes[i - 1] == 0xff && bytes[i] == 0xd9) {jpeg_size = i + 1; break;}
          }
        }
        if (jpeg_size) {publish_frame(bytes, jpeg_size); ++compressed_count_;}
        else RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid Gemini MJPEG frame");
      }
      if (retry_ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
        const int error = errno;
        if (error == ENODEV || error == ENXIO || error == EBADF) {
          RCLCPP_ERROR(
            get_logger(), "Gemini device disconnected during requeue: %s", std::strerror(error));
          device_lost = true;
          break;
        }
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000, "Gemini requeue failed: %s", std::strerror(error));
      }
      const auto current = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(current - last_report).count();
      if (elapsed >= 5.0) {
        RCLCPP_INFO(
          get_logger(), "Published compressed %.2f FPS, raw %.2f FPS",
          compressed_count_ / elapsed, raw_count_ / elapsed);
        compressed_count_ = 0;
        raw_count_ = 0;
        last_report = current;
      }
    }
    if (device_lost) {
      running_ = false;
      RCLCPP_ERROR(get_logger(), "Gemini capture stopped because the camera is unavailable");
      rclcpp::shutdown();
    }
  }

  int fd_ = -1;
  std::string device_, compressed_topic_, raw_topic_, frame_id_;
  int width_, height_, fps_, raw_preview_fps_;
  std::vector<Buffer> buffers_;
  std::atomic<bool> running_{false};
  std::thread worker_;
  std::chrono::steady_clock::time_point last_raw_publish_{};
  uint64_t compressed_count_ = 0;
  uint64_t raw_count_ = 0;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<Gemini2MjpegNode>());
  } catch (const std::exception & error) {
    fprintf(stderr, "FATAL: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
