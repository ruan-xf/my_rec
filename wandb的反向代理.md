

测试wandb可用性见 test_wandb.py  

开的代理也还是用不了，就考虑反向代理了  
代理的形式与 [GitHub加速下载代理 - 快速访问 GitHub 文件](https://gh-proxy.com/) 加前缀转换链接的用法一致  

- [使用 Cloudflare Workers 搭建反代 - 哔哩哔哩](https://www.bilibili.com/read/cv34109631/?opus_fallback=1)

域名一般需要买，免费的得找一找  
- [域名 rxf-proxy.filegear-sg.me. 注册 - L53](https://customer.l53.net/order/service/manage/136079)
- [rxf-proxy.filegear-sg.me | rxf-proxy.filegear-sg.me | Rxf7a6f@163.com's Account | Cloudflare](https://dash.cloudflare.com/449bbc48c69e8145934748e56226510f/rxf-proxy.filegear-sg.me)


将得到的域名添加到cf  
- [域名DNS服务托管至Cloudflare，就是如此简单_服务器_什么值得买](https://post.smzdm.com/p/a2xok2r2/)
- [Cloudflare托管域名，免费CDN加速，免费申请有效期15年的证书 - 知乎](https://zhuanlan.zhihu.com/p/650354462)



https://rxf-proxy.filegear-sg.me/https://wandb.ai  
跨域问题  


WANDB_BASE_URL=https://rxf-proxy.filegear-sg.me/https://api.wandb.ai  
数据没传上去，日志中出现403  
单独再用命令又能传上  

