# 朝阳升级说明

这个目录是可继续开发和部署的修复版：

- 启动入口：`app.py`
- 前端页面：`templates/index.html`
- 前端脚本：`static/js/game.js`
- 前端样式：`static/css/style.css`
- 依赖：`requirements.txt`
- 容器部署：`Dockerfile`

## 本地运行

1. 安装依赖  
   `pip install -r requirements.txt`

2. 启动服务  
   `python app.py`

3. 打开浏览器  
   `http://127.0.0.1:9999`

## 分享给朋友

1. 创建房间
2. 点击“复制邀请链接”
3. 把链接发给朋友

链接格式类似：

`http://你的域名:9999/?room=0001&name=张三`

## 部署建议

最省事的方式是把这个目录部署到支持 Docker 的平台，例如 Railway、Render、Fly.io 或你自己的云服务器。

核心要求只有两点：

- 对外开放 `9999` 端口，或由平台转发到公网
- 使用 `python app.py` 作为启动命令

## 当前已补的关键能力

- 房间创建、加入、选座、邀请链接
- WebSocket 实时同步
- 亮主、反亮、不亮
- 扣牌
- 换牌
- 出牌和聊天

## 仍建议后续补强

- 断线重连
- 房间持久化
- 开局权限控制
- 更严格的前端交互校验
