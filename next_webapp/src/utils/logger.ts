import winston from 'winston';

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

// Clean format for database and file logs.
const cleanFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss:ms' }),
  winston.format.printf(
    (info) => `${info.timestamp} ${info.level}: ${info.message}`,
  ),
);

// Colour is applied only to console output.
const consoleFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss:ms' }),
  winston.format.colorize({ all: true }),
  winston.format.printf(
    (info) => `${info.timestamp} ${info.level}: ${info.message}`,
  ),
);

const transports: winston.transport[] = [
  new winston.transports.Console({
    level: process.env.NODE_ENV === 'production' ? 'error' : 'debug',
    format: consoleFormat,
  }),
];

// MongoDB and file transports require the Node runtime.
// Edge middleware keeps console logging only.
if (process.env.NEXT_RUNTIME !== 'edge') {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const DatabaseTransport = require('./databaseTransport').default;

  transports.push(
    new DatabaseTransport({
      level: 'info',
      format: cleanFormat,
    }),
  );

  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const nodePath = require('path') as typeof import('path');

  transports.push(
    new winston.transports.File({
      filename: nodePath.join(process.cwd(), 'logs', 'error.log'),
      level: 'error',
      format: cleanFormat,
    }),
    new winston.transports.File({
      filename: nodePath.join(process.cwd(), 'logs', 'all.log'),
      format: cleanFormat,
    }),
  );
}

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  levels,
  transports,
});

export default logger;