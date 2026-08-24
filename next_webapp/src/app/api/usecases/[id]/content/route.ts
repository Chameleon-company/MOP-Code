import { NextRequest, NextResponse } from "next/server";
import { Readable } from "stream";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import UseCase from "@/models/mongoose/UseCase";
import { getGridFSBucket } from "@/lib/gridfs";
import { errorResponse } from "@/app/api/library/errorResponse";

// GridFS/streaming needs the Node runtime (mongoose, node:stream) — not edge.
export const runtime = "nodejs";

// content_type on the doc -> the Content-Type served for the streamed body.
// "notebook" bodies are the raw .ipynb JSON text; "html" is the legacy
// pre-migration HTML body.
const CONTENT_TYPE_BY_FORMAT: Record<string, string> = {
  notebook: "application/json; charset=utf-8",
  html: "text/html; charset=utf-8",
};

// The UseCase model is exported as `mongoose.models.UseCase || mongoose.model(...)`
// (see src/models/mongoose/UseCase.ts), which widens its static type to
// Model<any> — .lean() resolves to an untyped blob as a result. Cast to just
// the two fields this route actually reads instead of trusting inference.
interface UseCaseContentFields {
  content_file_id: mongoose.Types.ObjectId | null;
  content_type: "notebook" | "html" | null;
}

// ==============================
// GET /api/usecases/:id/content
// Streams the notebook/HTML body for a use case out of GridFS (PUBLIC —
// same read-visibility as GET /api/usecases/:id).
// ==============================
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return errorResponse("Invalid use case ID", 400, "INVALID_ID", request);
    }

    await dbConnect();

    const useCase = (await UseCase.findById(id).lean()) as UseCaseContentFields | null;
    if (!useCase) {
      return errorResponse("Use case not found", 404, "NOT_FOUND", request);
    }
    if (!useCase.content_file_id) {
      return errorResponse(
        "This use case has no notebook content",
        404,
        "NO_CONTENT",
        request,
      );
    }

    const bucket = await getGridFSBucket();

    // Confirm the file actually exists in GridFS before opening a download
    // stream / committing response headers. A doc can have content_file_id
    // set while the underlying file is missing (deleted out of band, a
    // failed upload never finishing, etc.) — check first so that case comes
    // back as a clean 404 instead of a 200 that errors mid-stream.
    const fileDoc = await bucket.find({ _id: useCase.content_file_id }).next();
    if (!fileDoc) {
      console.error(
        `[GET /api/usecases/${id}/content] content_file_id ${useCase.content_file_id} is set on the doc but missing in GridFS`,
      );
      return errorResponse(
        "Notebook content is missing",
        404,
        "FILE_NOT_FOUND",
        request,
      );
    }

    const downloadStream = bucket.openDownloadStream(useCase.content_file_id);
    downloadStream.on("error", (err) => {
      // File existed a moment ago (checked above) but the read still failed
      // (e.g. a chunk document is missing/corrupted). Headers may already
      // be committed by the time this fires, so all we can do is log —
      // Readable.toWeb already propagates this into the response stream so
      // the client sees a terminated/errored body rather than hanging.
      console.error(`[GET /api/usecases/${id}/content] GridFS read error:`, err);
    });

    const contentType =
      CONTENT_TYPE_BY_FORMAT[useCase.content_type ?? ""] ??
      "application/octet-stream";

    return new NextResponse(Readable.toWeb(downloadStream) as ReadableStream, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        // Moderate, safe default — nothing in the write path (yet) guarantees
        // content_file_id is immutable per-doc, so this stops short of a
        // long-lived/"immutable" cache policy until a later commit can
        // confirm edits always mint a new file id rather than overwriting.
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (error) {
    console.error("Get Use Case Content Error:", error);
    return errorResponse("Internal Server Error", 500, "INTERNAL_ERROR", request);
  }
}
