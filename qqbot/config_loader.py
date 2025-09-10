from dynaconf import Dynaconf

# 初始化 dynaconf 配置对象
settings = Dynaconf(
    settings_files=['config.yaml'],
    environments=False,  # 禁用环境分割
    load_dotenv=True,
    merge_enabled=True,
    # 支持环境变量覆盖，格式为 QQBOT_KEY_NAME
    envvar_prefix='QQBOT',
)

