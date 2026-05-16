import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { getAuthUser } from '@/app/api/library/auth';

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
  let filterUserIds: number[] | null = null;
  if (search) {
    const [{ data: detailsMatch }, { data: emailMatch }] = await Promise.all([
      supabase
        .from('user_details')
        .select('user_id')
        .or(`first_name.ilike.%${search}%,last_name.ilike.%${search}%`),
      supabase
        .from('user')
        .select('id')
        .ilike('email', `%${search}%`),
    ]);

    const ids = new Set<number>();
    detailsMatch?.forEach((d) => ids.add(d.user_id));
    emailMatch?.forEach((u)   => ids.add(u.id));
    filterUserIds = Array.from(ids);

    if (filterUserIds.length === 0) {
      const emptyPagination = { page, pageSize, total: 0, totalPages: 0 };
      if (format === 'csv') return new NextResponse('id,activity,performedBy,performedAt\n', {
        headers: { 'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename="activity-history.csv"' },
      });
      return NextResponse.json({ success: true, data: [], pagination: emptyPagination });
    }
  }

  // Build base query
  let query = supabase
    .from('logs')
    .select('id, method, url, user_id, timestamp', { count: 'exact' })
    .eq('source', 'middleware')
    .not('user_id', 'is', null)
    .order(sortBy, { ascending });

  if (!includeGet) {
    query = query.in('method', ['POST', 'PUT', 'PATCH', 'DELETE']);
  }

  if (filterUserIds !== null) {
    query = query.in('user_id', filterUserIds);
  }

  // For CSV, fetch all records; for JSON, paginate
  if (format !== 'csv') {
    query = query.range(offset, offset + pageSize - 1);
  }

  const { data: logs, error, count } = await query;

  if (error) {
    return NextResponse.json({ success: false, message: 'Failed to fetch activity' }, { status: 500 });
  }

  // Resolve user display names
  const userIds = [...new Set((logs ?? []).map((l) => l.user_id).filter(Boolean))] as number[];
  const userNameMap: Record<number, string> = {};

  if (userIds.length > 0) {
    const [{ data: details }, { data: users }] = await Promise.all([
      supabase.from('user_details').select('user_id, first_name, last_name').in('user_id', userIds),
      supabase.from('user').select('id, email').in('id', userIds),
    ]);

    const emailMap: Record<number, string> = {};
    users?.forEach((u) => { emailMap[u.id] = u.email; });

    details?.forEach((d) => {
      const name = `${d.first_name ?? ''} ${d.last_name ?? ''}`.trim();
      userNameMap[d.user_id] = name || emailMap[d.user_id] || `User #${d.user_id}`;
    });

    userIds.forEach((id) => {
      if (!userNameMap[id]) userNameMap[id] = emailMap[id] || `User #${id}`;
    });
  }

  const mapped = (logs ?? []).map((log) => ({
    id: log.id,
    activity:    formatActivity(log.method, log.url),
    performedBy: log.user_id ? (userNameMap[log.user_id] ?? `User #${log.user_id}`) : 'System',
    performedAt: log.timestamp,
  }));

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
  const total = count ?? 0;
  return NextResponse.json({
    success: true,
    data: mapped,
    pagination: { page, pageSize, total, totalPages: Math.ceil(total / pageSize) },
  });
}
