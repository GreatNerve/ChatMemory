export function formatEta(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) {
    return null;
  }
  const total = Math.round(seconds);
  if (total < 60) {
    return `~${total}s left`;
  }
  const minutes = Math.floor(total / 60);
  const remSec = total % 60;
  if (minutes < 60) {
    return remSec > 0 ? `~${minutes}m ${remSec}s left` : `~${minutes}m left`;
  }
  const hours = Math.floor(minutes / 60);
  const remMin = minutes % 60;
  return remMin > 0 ? `~${hours}h ${remMin}m left` : `~${hours}h left`;
}
