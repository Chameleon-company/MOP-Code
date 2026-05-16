module.exports = {
    testEnvironment: 'jsdom',
    setupFilesAfterEnv: ['@testing-library/jest-dom/extend-expect'],
    transform: {
      "^.+\\.jsx?$": "babel-jest",
    },
    babelConfig: require('./babel.config.js'),
  };