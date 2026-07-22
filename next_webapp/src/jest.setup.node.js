// src/jest.setup.node.js
const { Request, Response, Headers, fetch } = require('node-fetch');

global.Request  = Request;
global.Response = Response;
global.Headers  = Headers;
global.fetch    = fetch;