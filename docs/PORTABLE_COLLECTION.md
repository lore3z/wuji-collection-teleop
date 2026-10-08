# 安装与迁移

安装、采集、读取及设备配置的完整说明统一维护在 [README.md](../README.md)。

克隆源码后运行 `./scripts/setup.sh --install-system`，复制 `config/local.env.example` 为
`config/local.env` 并填写本机设备。不要复制 `.venv/`、`.runtime/`、驱动的
`build/install/log/`；目标机器重新构建。数据、个人 SDK 标定和模型工程单独迁移。

导出当前已提交源码（不含本机文件）：

```bash
./tools/build_portable_bundle.sh /目标路径/wuji-collection-teleop
```

FTP-1 质量合同见 [FTP1_COLLECTION_QUALITY.md](FTP1_COLLECTION_QUALITY.md)。
