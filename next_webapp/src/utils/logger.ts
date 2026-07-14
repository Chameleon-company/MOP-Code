import winston from 'winston';
import DatabaseTransport from './databaseTransport';

const levels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  debug: 4,
};

const colors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  http: 'magenta',
  debug: 'white',
};

winston.addColors(colors);

const format = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss:ms' }),
  winston.format.colorize({ all: true }),
  winston.format.printf(
    (info) => `${info.timestamp} ${info.level}: ${info.message}`,
  ),
);

const transports: winston.transport[] = [
  new winston.transports.Console({
    level: process.env.NODE_ENV === 'production' ? 'error' : 'debug',
  }),
  new DatabaseTransport({
    level: 'info',
  }),
];

// File transports need Node.js `path` and `fs` — not available in Edge Runtime.
// `process.env.NEXT_RUNTIME` is replaced at build time, so webpack dead-code-eliminates
// this entire block (including the require) from the Edge bundle.
if (process.env.NEXT_RUNTIME !== 'edge') {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const nodePath = require('path') as typeof import('path');
  transports.push(
    new winston.transports.File({
      filename: nodePath.join(process.cwd(), 'logs', 'error.log'),
      level: 'error',
    }),
    new winston.transports.File({
      filename: nodePath.join(process.cwd(), 'logs', 'all.log'),
    }),
  );
}

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  levels,
  format,
  transports,
});

export default logger;