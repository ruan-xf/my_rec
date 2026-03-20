#!/usr/bin/env python
"""独立的 wandb 功能测试脚本"""
import os
import wandb
import random
from datetime import datetime


import logging
logging.basicConfig(level=logging.DEBUG)

# 可配置的测试选项
TEST_CONFIG = {
    "mode": "online",  # 可选: online, offline, disabled
    "project": "test-wandb-connection",
    "run_name": f"test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "num_steps": 10,
}

def test_wandb_connection():
    print(f"开始测试 wandb，模式: {TEST_CONFIG['mode']}")

    # 初始化 wandb
    try:
        os.environ["WANDB__SERVICE_WAIT"]="90"
        run = wandb.init(
            project=TEST_CONFIG["project"],
            name=TEST_CONFIG["run_name"],
            mode=TEST_CONFIG["mode"],
            config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "architecture": "test-model",
            }
        )
        print(f"✅ wandb 初始化成功！Run ID: {run.id}")
        if TEST_CONFIG["mode"] == "online":
            print(f"🔗 运行页面: {run.url}")
    except Exception as e:
        print(f"❌ wandb 初始化失败: {str(e)}")
        return False

    # 测试记录指标
    print(f"\n开始记录 {TEST_CONFIG['num_steps']} 步指标...")
    try:
        for step in range(TEST_CONFIG["num_steps"]):
            # 模拟训练指标
            train_loss = random.random() * (1 - step/TEST_CONFIG["num_steps"])
            val_loss = random.random() * (1.2 - step/TEST_CONFIG["num_steps"])
            accuracy = 0.5 + random.random() * (step/TEST_CONFIG["num_steps"])

            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/accuracy": accuracy,
                "step": step
            }, step=step)

            if (step + 1) % 5 == 0:
                print(f"   已记录 {step + 1}/{TEST_CONFIG['num_steps']} 步")

        # 测试记录表格
        table = wandb.Table(columns=["step", "metric", "value"])
        for i in range(5):
            table.add_data(i, "test_metric", random.random())
        wandb.log({"test_table": table})

        print("✅ 所有指标记录成功")

    except Exception as e:
        print(f"❌ 记录指标失败: {str(e)}")
        wandb.finish(exit_code=1)
        return False

    # 结束运行
    try:
        wandb.finish()
        print("\n✅ wandb 测试完成！")
        if TEST_CONFIG["mode"] == "offline":
            print("💡 离线模式下，运行结果保存在本地 ./wandb 目录，使用 'wandb sync' 命令同步")
        return True
    except Exception as e:
        print(f"❌ 结束 wandb 运行失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 先检查环境变量
    print("环境变量检查:")
    print(f"WANDB_API_KEY: {'已设置' if os.getenv('WANDB_API_KEY') else '未设置'}")
    print(f"WANDB_MODE: {os.getenv('WANDB_MODE', '默认(online)')}")
    print(f"WANDB_PROJECT: {os.getenv('WANDB_PROJECT', '未设置')}\n")

    success = test_wandb_connection()

    if not success:
        print("\n🔍 排查建议:")
        print("1. 先运行 `wandb login` 确认已正确配置 API Key")
        print("2. 检查网络连接和代理设置，确保可以访问 wandb.ai")
        print("3. 尝试使用离线模式测试: 修改 TEST_CONFIG['mode'] = 'offline'")
        print("4. 查看 wandb 日志: ~/.wandb/ 目录下的日志文件")
        exit(1)