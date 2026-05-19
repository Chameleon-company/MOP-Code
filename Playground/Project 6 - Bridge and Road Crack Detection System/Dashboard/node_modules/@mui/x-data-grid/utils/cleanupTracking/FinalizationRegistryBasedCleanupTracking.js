"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.FinalizationRegistryBasedCleanupTracking = void 0;
class FinalizationRegistryBasedCleanupTracking {
  registry = new FinalizationRegistry(unsubscribe => {
    if (typeof unsubscribe === 'function') {
      unsubscribe();
    }
  });
  register(object, unsubscribe, unregisterToken) {
    this.registry.register(object, unsubscribe, unregisterToken);
  }
  unregister(unregisterToken) {
    this.registry.unregister(unregisterToken);
  }
  reset() {}
}
exports.FinalizationRegistryBasedCleanupTracking = FinalizationRegistryBasedCleanupTracking;