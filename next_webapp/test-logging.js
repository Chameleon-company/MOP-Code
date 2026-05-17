// Test script for database logging
import logger from './src/utils/logger.js';

async function testLogging() {
  console.log('Testing database logging...');

  // Test different log levels
  logger.info('Test info message', {
    source: 'test',
    user_id: 1,
    ip_address: '127.0.0.1',
  });

  logger.error('Test error message', {
    source: 'test',
    user_id: 1,
    status_code: 500,
  });

  logger.warn('Test warning message', {
    source: 'test',
  });

  console.log('Logging test completed. Check database for entries.');
}

testLogging().catch(console.error);