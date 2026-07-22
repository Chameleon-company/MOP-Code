export declare function useRunOncePerLoop<T extends (...args: any[]) => void>(callback: T): {
  schedule: (...args: Parameters<T>) => void;
  cancel: () => boolean;
};