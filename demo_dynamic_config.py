#!/usr/bin/env python3
"""
动态配置演示脚本
展示如何在运行时更新配置而无需重启应用
"""

import time
import yaml
from qqbot.config_loader import settings

def update_config_file(key: str, value):
    """更新配置文件中的值"""
    config_file = 'config.yaml'
    
    # 读取当前配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config[key] = value
    
    # 写回文件
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ 已更新配置文件: {key} = {value}")

def demo_dynamic_config():
    """演示动态配置更新"""
    print("=== 动态配置更新演示 ===\n")
    
    # 显示当前配置
    print("当前配置:")
    print(f"  - bot_name: {settings.bot_name}")
    print(f"  - enable_vision: {settings.enable_vision}")
    print(f"  - max_messages_history: {settings.max_messages_history}")
    print()
    
    # 更新配置文件
    print("更新配置文件...")
    update_config_file('bot_name', 'Lain-V2')
    update_config_file('enable_vision', True)
    update_config_file('max_messages_history', 1000)
    print()
    
    # 手动重新加载配置
    print("重新加载配置...")
    settings.reload()
    time.sleep(0.1)  # 短暂等待文件系统更新
    
    print("重新加载后的配置:")
    print(f"  - bot_name: {settings.bot_name}")
    print(f"  - enable_vision: {settings.enable_vision}")
    print(f"  - max_messages_history: {settings.max_messages_history}")
    print()
    
    # 恢复原始配置
    print("恢复原始配置...")
    update_config_file('bot_name', 'Lain')
    update_config_file('enable_vision', False)
    update_config_file('max_messages_history', 800)
    print()
    
    # 再次重新加载
    settings.reload()
    time.sleep(0.1)
    print("恢复后的配置:")
    print(f"  - bot_name: {settings.bot_name}")
    print(f"  - enable_vision: {settings.enable_vision}")
    print(f"  - max_messages_history: {settings.max_messages_history}")
    
    print("\n✓ 动态配置更新演示完成！")

if __name__ == "__main__":
    demo_dynamic_config()
