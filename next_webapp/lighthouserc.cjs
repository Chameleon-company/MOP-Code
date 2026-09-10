module.exports = {
  ci: {
    collect: {
      startServerCommand: 'npm run start -- --hostname localhost --port 3001',
      startServerReadyPattern: 'Ready in',
      startServerReadyTimeout: 60000,
      url: ['http://localhost:3001/en', 'http://localhost:3001/en/about', 'http://localhost:3001/en/privacypolicy'],
      numberOfRuns: 3,
      settings: { chromeFlags: '--headless --no-sandbox --disable-dev-shm-usage' },
    },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.7, aggregationMethod: 'median' }],
        'largest-contentful-paint': ['error', { maxNumericValue: 4000, aggregationMethod: 'median' }],
        'total-blocking-time': ['error', { maxNumericValue: 600, aggregationMethod: 'median' }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1, aggregationMethod: 'median' }],
        'http-status-code': 'error',
      },
    },
    upload: { target: 'filesystem', outputDir: './lighthouse-reports' },
  },
};
