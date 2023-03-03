const PORT = 8000;
const app = require("./app");
const http = require("http");

const server = http.createServer(app);

async function startServer() {
  server.listen(PORT, () => {
    console.log(`Listening on port ${PORT}...`);
  });
}

startServer();