/**
 * Helpers for the spend row CheckVideoCost writes when a video job finishes.
 *
 * That row's request id is "<video_id>_video_cost" and its call type is avideo_retrieve.
 * The video id itself stays the caller's id; the suffix only keeps the row from colliding
 * with the create log.
 */

export const VIDEO_COST_REQUEST_ID_SUFFIX = "_video_cost";

const VIDEO_COST_CALL_TYPES = ["avideo_retrieve", "video_retrieve"];

/** The caller-facing video id behind a poller-written "<video_id>_video_cost" spend row. */
export const getVideoIdFromCostRequestId = (callType: string, requestId: string): string | undefined => {
  if (!VIDEO_COST_CALL_TYPES.includes(callType)) return undefined;
  if (!requestId.endsWith(VIDEO_COST_REQUEST_ID_SUFFIX) || requestId.length <= VIDEO_COST_REQUEST_ID_SUFFIX.length) {
    return undefined;
  }
  return requestId.slice(0, -VIDEO_COST_REQUEST_ID_SUFFIX.length);
};
