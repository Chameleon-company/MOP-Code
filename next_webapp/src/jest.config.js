// src/jest.config.js
const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: './', // root of your Next.js app (where next.config.js lives)
});

const customConfig = {
  setupFilesAfterEnv: ['@testing-library/jest-dom'],

  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },

  testMatch: [
    '**/__tests__/**/*.(test|spec).(ts|tsx|js|jsx)',
  ],

  // no testEnvironment here — next/jest sets it correctly per file
};

module.exports = createJestConfig(customConfig);