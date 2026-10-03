// Frame timer in a Web Worker: background tabs throttle main-thread timers to ~1 Hz, worker timers far less
// (master spec 5.6). Post an interval in ms to start, 0 to stop; it posts back one message per tick.
let id: ReturnType<typeof setInterval> | undefined;
self.onmessage = (e: MessageEvent<number>) => {
  clearInterval(id);
  if (e.data > 0) id = setInterval(() => self.postMessage(0), e.data);
};
