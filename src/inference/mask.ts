/** Write IEEE-754 half bits, regardless of whether ORT exposes Float16Array or Uint16Array. */
export function writeMask(
  storage: ArrayBufferView,
  channels: number,
  disabled: number[],
) {
  if (storage.byteLength !== channels * 2)
    throw new Error('Invalid half-precision mask storage')
  if (disabled.some((c) => !Number.isInteger(c) || c < 0 || c >= channels))
    throw new Error('Invalid channel index')
  const bits = new Uint16Array(storage.buffer, storage.byteOffset, channels)
  bits.fill(0x3c00)
  for (const channel of disabled) bits[channel] = 0
}
