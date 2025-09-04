# Makefile for QQ Bot Docker Management

# 变量定义
IMAGE_NAME = lain-bot
CONTAINER_NAME = lain
VERSION = $(shell date +%Y%m%d-%H%M%S)
LATEST_TAG = $(IMAGE_NAME):latest
VERSION_TAG = $(IMAGE_NAME):$(VERSION)

.PHONY: deploy
deploy:
	@echo "开始部署 QQ Bot..."
	@echo "版本: $(VERSION)"
	@echo "1. 构建新镜像..."
	docker build -t $(LATEST_TAG) -t $(VERSION_TAG) .
	@echo "2. 停止并删除旧容器..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@echo "3. 运行新容器..."
	docker run -d --name $(CONTAINER_NAME) --restart unless-stopped --network="host" $(LATEST_TAG)
	@echo "4. 清理旧镜像（保留最近3个版本）..."
	@docker images $(IMAGE_NAME) --format "{{.Tag}}" | grep -v latest | tail -n +4 | xargs -I {} docker rmi $(IMAGE_NAME):{} 2>/dev/null || true
	@echo "清理完成，保留最近3个版本"
	@echo "部署完成！"
	@echo "当前版本: $(VERSION)"
	@echo "使用 'make logs' 查看运行日志"
	@echo "使用 'make stop' 停止容器"
	@echo "使用 'make list-versions' 查看所有版本"

# 查看容器日志
.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME)

# 停止容器
.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME)

# 查看所有镜像版本
.PHONY: list-versions
list-versions:
	@echo "所有镜像版本:"
	docker images $(IMAGE_NAME) --format "table {{.Tag}}\t{{.CreatedAt}}\t{{.Size}}" | head -10

# 回滚到指定版本
.PHONY: rollback
rollback:
	@echo "可用版本:"
	docker images $(IMAGE_NAME) --format "{{.Tag}}" | grep -v latest | head -5
	@echo "使用方法: make rollback-to VERSION=版本号"
	@echo "例如: make rollback-to VERSION=20241201-143022"

# 回滚到指定版本
.PHONY: rollback-to
rollback-to:
	@if [ -z "$(VERSION)" ]; then \
		echo "请指定版本号，例如: make rollback-to VERSION=20241201-143022"; \
		exit 1; \
	fi
	@echo "回滚到版本: $(VERSION)"
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	docker run -d --name $(CONTAINER_NAME) --restart unless-stopped --network="host" $(IMAGE_NAME):$(VERSION)
	@echo "回滚完成！"

# 清理旧版本镜像（保留最近3个版本）
.PHONY: cleanup-versions
cleanup-versions:
	@echo "清理旧版本镜像..."
	@docker images $(IMAGE_NAME) --format "{{.Tag}}" | grep -v latest | tail -n +4 | xargs -I {} docker rmi $(IMAGE_NAME):{} 2>/dev/null || true
	@echo "清理完成，保留最近3个版本"
