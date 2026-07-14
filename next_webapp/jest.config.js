// jest.config.js  ← ROOT level, next to package.json
const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: './',  // finds next.config.js and .env here
});

module.exports = createJestConfig({
  setupFilesAfterEnv: ['@testing-library/jest-dom'],

  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },

  testMatch: [
    '**/src/__tests__/**/*.(test|spec).(ts|tsx|js|jsx)',
  ],
});