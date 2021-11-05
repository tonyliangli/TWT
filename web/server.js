const express = require('express');
const timeout = require('connect-timeout');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();

// Local host and port
const HOST = '0.0.0.0', PORT = 4201;

// Remote API host
const API_HOST = 'http://localhost:5001'

// Time out
// const TIME_OUT = 30 * 1e3;

// Set time out
// app.use(timeout(TIME_OUT));
// app.use((req, res, next) => {
//   if (!req.timedout) next();
// });

// Set static resources
app.use('/', express.static('public'));

// Reverse proxy
// e.g.: proxy /api to ${API_HOST}/api
// app.use(createProxyMiddleware('/api', { target: API_HOST }));
// Custom proxy rules
app.use(createProxyMiddleware('/api', {
  target: API_HOST, // target host
  changeOrigin: true, // needed for virtual hosted sites
  ws: true, // proxy web sockets
  // pathRewrite: {
    // '^/api': '', // rewrite path
  // }
}));

// List to host and port
var server = app.listen(PORT, HOST, () => {
  var host = server.address().address
  var port = server.address().port
  console.log("server running at http://%s:%s", host, port)
});
server.setTimeout(0)
