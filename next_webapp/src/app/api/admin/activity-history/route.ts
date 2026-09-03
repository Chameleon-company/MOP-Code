import { NextRequest, NextResponse } from 'next/server';
import dbConnect from "@/lib/dbConnect";
import Log from "@/models/mongoose/Log";
import User from "@/models/mongoose/User";
import { getAuthUser } from '@/app/api/library/auth';

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function formatActivity(method: string | null, url: string | null): string {
  if (!method || !url) return 'Unknown action';
  const path = url.split('?')[0];
  const m = method.toUpperCase();

  const blogMatch    = path.match(/^\/api\/blogs\/(\w+)$/);
  const ucMatch      = path.match(/^\/api\/usecases\/(\w+)$/);
  const galleryMatch = path.match(/^\/api\/gallery\/(\w+)$/);
  const catMatch     = path.match(/^\/api\/categories\/(\w+)$/);

  if (m === 'POST'   && path === '/api/blogs')    return 'Added a blog';
  if (m === 'PUT'    && blogMatch)                return `Updated blog #${blogMatch[1]}`;
  if (m === 'DELETE' && blogMatch)                return `Deleted blog #${blogMatch[1]}`;

  if (m === 'POST'   && path === '/api/upload')   return 'Uploaded a use case';
  if (m === 'PUT'    && ucMatch)                  return `Updated use case #${ucMatch[1]}`;
  if (m === 'DELETE' && ucMatch)                  return `Deleted use case #${ucMatch[1]}`;

  if (m === 'POST'   && path === '/api/gallery')  return 'Added a gallery image';
  if (m === 'PUT'    && galleryMatch)             return `Updated gallery image #${galleryMatch[1]}`;
  if (m === 'DELETE' && galleryMatch)             return `Deleted gallery image #${galleryMatch[1]}`;

  if (m === 'POST'   && path === '/api/categories') return 'Added a category';
  if (m === 'PUT'    && catMatch)                   return `Updated category #${catMatch[1]}`;
  if (m === 'DELETE' && catMatch)                   return `Deleted category #${catMatch[1]}`;

  if (m === 'PUT'    && path === '/api/profile')                return 'Updated profile';
  if (m === 'POST'   && path === '/api/profile/upload-image')   return 'Updated profile picture';

  if (m === 'GET') return `Viewed ${path}`;

  return `${m} ${path}`;
}

const ALLOWED_SORT_COLUMNS = new Set(['timestamp']);

export async function GET(request: NextRequest) {
  const { isAuthenticated, isAdmin } = getAuthUser(request);
  if (!isAuthenticated) {
    return NextResponse.json({ success: false, message: 'Unauthorized' }, { status: 401 });
  }
  if (!isAdmin) {
    return NextResponse.json({ success: false, message: 'Forbidden' }, { status: 403 });
  }

  try {
    await dbConnect();
    const url   = new URL(request.url);
    const sp    = url.searchParams;

    const format     = sp.get('format') ?? 'json';           // 'json' | 'csv'
    const includeGet = sp.get('includeGet') === 'true';       // default: write-ops only
    const search     = sp.get('search')?.trim() ?? '';

    // Sort
    const rawSortBy    = sp.get('sortBy') ?? 'timestamp';
    const rawSortOrder = sp.get('sortOrder') ?? 'desc';
    const sortBy       = ALLOWED_SORT_COLUMNS.has(rawSortBy) ? rawSortBy : 'timestamp';
    const ascending    = rawSortOrder.toLowerCase() === 'asc';

    // Pagination (ignored for CSV export — returns all rows)
    const page     = Math.max(1, parseInt(sp.get('page')     || '1'));
    const pageSize = Math.min(50, parseInt(sp.get('pageSize') || '20'));
    const offset   = (page - 1) * pageSize;

    // Resolve search term → matching user_ids
    let filterUserIds: string[] | null = null;
    if (search) {
      const escapedSearch = escapeRegex(search);

      const matchingUsers = await User.find({
        $or: [
          { email: { $regex: escapedSearch, $options: "i" } },
          { "profile.first_name": { $regex: escapedSearch, $options: "i" } },
          { "profile.last_name": { $regex: escapedSearch, $options: "i" } },
        ],
      })
        .select("_id")
        .lean();

      filterUserIds = matchingUsers.map((u) => u._id.toString());

      if (filterUserIds.length === 0) {
        const emptyPagination = {
          page,
          pageSize,
          total: 0,
          totalPages: 0,
        };

        if (format === "csv") {
          return new NextResponse(
            "id,activity,performedBy,performedAt\n",
            {
              headers: {
                "Content-Type": "text/csv",
                "Content-Disposition":
                  'attachment; filename="activity-history.csv"',
              },
            }
          );
        }

        return NextResponse.json({
          success: true,
          data: [],
          pagination: emptyPagination,
        });
      }
    }

    // Build base query
    const logFilter: Record<string, unknown> = {
      source: "middleware",
      user_id: { $ne: null },
    };

    if (!includeGet) {
      logFilter.method = {
        $in: ["POST", "PUT", "PATCH", "DELETE"],
      };
    }

    if (filterUserIds !== null) {
      logFilter.user_id = {
        $in: filterUserIds,
      };
    }

    const sortDirection = ascending ? 1 : -1;

    let logsQuery = Log.find(logFilter)
      .sort({ [sortBy]: sortDirection });

    if (format !== "csv") {
      logsQuery = logsQuery
        .skip(offset)
        .limit(pageSize);
    }

    const [logs, total] = await Promise.all([
      logsQuery.lean(),
      Log.countDocuments(logFilter),
    ]);

    // Resolve user display names
    const userIds = [
      ...new Set(
        logs
          .map((l) => l.user_id?.toString())
          .filter((id): id is string => Boolean(id))
      ),
    ];

    const userNameMap: Record<string, string> = {};

    if (userIds.length > 0) {
        const users = await User.find({
          _id: { $in: userIds },
        })
          .select("_id email profile.first_name profile.last_name")
          .lean();

      users.forEach((user) => {
        const id = user._id.toString();
        const name =
          `${user.profile?.first_name ?? ""} ${user.profile?.last_name ?? ""}`.trim();userNameMap[id] =name ||user.email || `User #${id}`;
      });
    }

    const mapped = logs.map((log) => {
      const userId = log.user_id?.toString();

      return {
        id: log._id.toString(),
        activity: formatActivity(log.method, log.url),
        performedBy: userId
          ? userNameMap[userId] ?? `User #${userId}`
          : "System",
        performedAt: log.timestamp,
      };
    });

    // ── CSV export ────────────────────────────────────────────────────────────
    if (format === 'csv') {
      const escape = (v: string) => `"${String(v ?? '').replace(/"/g, '""')}"`;
      const rows = mapped.map((r) =>
        [r.id, escape(r.activity), escape(r.performedBy), r.performedAt].join(',')
      );
      const csv = ['id,activity,performedBy,performedAt', ...rows].join('\n');

      return new NextResponse(csv, {
        headers: {
          'Content-Type': 'text/csv',
          'Content-Disposition': 'attachment; filename="activity-history.csv"',
        },
      });
    }

    // ── JSON response ─────────────────────────────────────────────────────────
    return NextResponse.json({
      success: true,
      data: mapped,
      pagination: { page, pageSize, total, totalPages: Math.ceil(total / pageSize) },
    });
  }
  catch (error) {
      console.error("[GET /api/activity] error:", error);

      return NextResponse.json(
        { success: false, message: "Failed to fetch activity" },
        { status: 500 }
      );
    }
}
