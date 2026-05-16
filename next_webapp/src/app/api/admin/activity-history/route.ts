import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/library/supabaseClient';
import { getAuthUser } from '@/app/api/library/auth';

function formatActivity(method: string | null, url: string | null): string {
  if (!method || !url) return 'Unknown action';
  const path = url.split('?')[0];
  const m = method.toUpperCase();

  const blogMatch   = path.match(/^\/api\/blogs\/(\w+)$/);
  const ucMatch     = path.match(/^\/api\/usecases\/(\w+)$/);
  const galleryMatch = path.match(/^\/api\/gallery\/(\w+)$/);
  const catMatch    = path.match(/^\/api\/categories\/(\w+)$/);

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

  if (m === 'PUT'    && path === '/api/profile')  return 'Updated profile';
  if (m === 'POST'   && path === '/api/profile/upload-image') return 'Updated profile picture';

  return `${m} ${path}`;
}

export async function GET(request: NextRequest) {
  const { isAuthenticated, isAdmin } = getAuthUser(request);
  if (!isAuthenticated) {
    return NextResponse.json({ success: false, message: 'Unauthorized' }, { status: 401 });
  }
  if (!isAdmin) {
    return NextResponse.json({ success: false, message: 'Forbidden' }, { status: 403 });
  }

  const url = new URL(request.url);
  const page     = Math.max(1, parseInt(url.searchParams.get('page')     || '1'));
  const pageSize = Math.min(50, parseInt(url.searchParams.get('pageSize') || '20'));
  const search   = url.searchParams.get('search')?.trim() ?? '';
  const offset   = (page - 1) * pageSize;

  // If searching by user name/email, resolve matching user_ids first
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
    emailMatch?.forEach((u) => ids.add(u.id));
    filterUserIds = Array.from(ids);

    if (filterUserIds.length === 0) {
      return NextResponse.json({
        success: true,
        data: [],
        pagination: { page, pageSize, total: 0, totalPages: 0 },
      });
    }
  }

  // Query admin action logs: middleware-logged, non-GET, with a user_id
  let query = supabase
    .from('logs')
    .select('id, method, url, user_id, timestamp', { count: 'exact' })
    .eq('source', 'middleware')
    .in('method', ['POST', 'PUT', 'PATCH', 'DELETE'])
    .not('user_id', 'is', null)
    .order('timestamp', { ascending: false })
    .range(offset, offset + pageSize - 1);

  if (filterUserIds !== null) {
    query = query.in('user_id', filterUserIds);
  }

  const { data: logs, error, count } = await query;

  if (error) {
    return NextResponse.json({ success: false, message: 'Failed to fetch activity' }, { status: 500 });
  }

  // Look up names for all user_ids in this page
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

  const total = count ?? 0;

  return NextResponse.json({
    success: true,
    data: (logs ?? []).map((log) => ({
      id: log.id,
      activity: formatActivity(log.method, log.url),
      performedBy: log.user_id ? (userNameMap[log.user_id] ?? `User #${log.user_id}`) : 'System',
      performedAt: log.timestamp,
    })),
    pagination: { page, pageSize, total, totalPages: Math.ceil(total / pageSize) },
  });
}
