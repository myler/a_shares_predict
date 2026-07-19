App({
  onLaunch() {
    // 初始化：检查缓存状态
    const info = wx.getSystemInfoSync();
    console.log('三合资本启动', info.model);
  },
});
