type Listener<T> = (value: T) => void;
export declare class Store<State> {
  state: State;
  private listeners;
  private updateTick;
  static create<T>(state: T): Store<T>;
  constructor(state: State);
  subscribe: (fn: Listener<State>) => () => void;
  /**
   * Returns the current state snapshot. Meant for usage with `useSyncExternalStore`.
   * If you want to access the state, use the `state` property instead.
   */
  getSnapshot: () => State;
  setState(newState: State): void;
  update(changes: Partial<State>): void;
  set<Key extends keyof State, T extends State[Key]>(key: Key, value: T): void;
  use: <F extends (...args: any) => any>(selector: F, ...args: SelectorArgs<F>) => ReturnType<F>;
}
export type ReadonlyStore<State> = Pick<Store<State>, 'getSnapshot' | 'subscribe' | 'state'>;
type SelectorArgs<Selector> = Selector extends ((...params: infer Params) => any) ? Tail<Params> : never;
type Tail<T extends readonly any[]> = T extends readonly [any, ...infer Rest] ? Rest : [];
export {};