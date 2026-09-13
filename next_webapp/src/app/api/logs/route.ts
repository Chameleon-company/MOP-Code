import { NextRequest, NextResponse } from 'next/server';
import { Types } from 'mongoose';
import dbConnect from '@/lib/dbConnect';
import Log from '@/models/mongoose/Log';
import User from '@/models/mongoose/User';
import { getAuthUser } from '@/app/api/library/auth';
import { errorResponse } from '@/app/api/library/errorResponse';
import logger from '@/utils/logger';

const ALLOWED_SORT = new Set([
  'timestamp',
  'level',
  'source',
  'method',
  'status_code',
  'response_time',
]);

function parsePositiveInteger(
  value: string | null,
  fallback: number,
): number {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function normalizeLegacyId(value: unknown): string | number | null {
  if (value === null || value === undefined) {
    return null;
  }

  const text = String(value);
  const numberValue = Number(text);

  return Number.isSafeInteger(numberValue) && String(numberValue) === text
    ? numberValue
    : text;
}

function getResponseUserId(user: any): string | number | null {
  if (!user) {
    return null;
  }

  if (typeof user === 'object' && user.legacy_id) {
    return normalizeLegacyId(user.legacy_id);
  }

  if (typeof user === 'object' && user._id) {
    return String(user._id);
  }

  return String(user);
}

function toLogDTO(log: any) {
  const {
    _id,
    __v,
    legacy_id: legacyId,
    user_id: userId,
    ...logData
  } = log;

  return {
    id: normalizeLegacyId(legacyId) ?? String(_id),
    ...logData,
    user_id: getResponseUserId(userId),
  };
}

// GET /api/logs
// Returns paginated logs for admin users.
export async function GET(request: NextRequest) {
  try {
    const authUser = getAuthUser(request);

    if (!authUser.isAuthenticated) {
      logger.warn('Unauthorized access attempt to logs API', {
        source: 'api',
        url: '/api/logs',
        ip_address:
          request.headers.get('x-forwarded-for') ||
          request.headers.get('x-real-ip') ||
          'unknown',
      });

      return NextResponse.json(
        { success: false, message: 'Unauthorized' },
        { status: 401 },
      );
    }

    if (!authUser.isAdmin) {
      logger.warn('Non-admin access attempt to logs API', {
        source: 'api',
        url: '/api/logs',
        ip_address:
          request.headers.get('x-forwarded-for') ||
          request.headers.get('x-real-ip') ||
          'unknown',
      });

      return NextResponse.json(
        { success: false, message: 'Forbidden - Admin only' },
        { status: 403 },
      );
    }

    const url = new URL(request.url);

    const page = parsePositiveInteger(url.searchParams.get('page'), 1);
    const pageSize = Math.min(
      parsePositiveInteger(url.searchParams.get('pageSize'), 50),
      200,
    );

    const level = url.searchParams.get('level');
    const source = url.searchParams.get('source');
    const startDate = url.searchParams.get('startDate');
    const endDate = url.searchParams.get('endDate');
    const search = url.searchParams.get('search');
    const rawUserId = url.searchParams.get('user_id');

    const rawSortBy = url.searchParams.get('sortBy') ?? 'timestamp';
    const rawSortOrder = url.searchParams.get('sortOrder') ?? 'desc';

    const sortBy = ALLOWED_SORT.has(rawSortBy)
      ? rawSortBy
      : 'timestamp';

    const sortDirection: 1 | -1 =
      rawSortOrder.toLowerCase() === 'asc' ? 1 : -1;

    const filter: Record<string, unknown> = {};

    if (level) {
      filter.level = level;
    }

    if (source) {
      filter.source = source;
    }

    if (search) {
      filter.message = {
        $regex: escapeRegex(search),
        $options: 'i',
      };
    }

    const timestampFilter: {
      $gte?: Date;
      $lte?: Date;
    } = {};

    if (startDate) {
      const parsedStartDate = new Date(startDate);

      if (Number.isNaN(parsedStartDate.getTime())) {
        return NextResponse.json(
          { success: false, message: 'Invalid startDate' },
          { status: 400 },
        );
      }

      timestampFilter.$gte = parsedStartDate;
    }

    if (endDate) {
      const parsedEndDate = new Date(endDate);

      if (Number.isNaN(parsedEndDate.getTime())) {
        return NextResponse.json(
          { success: false, message: 'Invalid endDate' },
          { status: 400 },
        );
      }

      timestampFilter.$lte = parsedEndDate;
    }

    if (Object.keys(timestampFilter).length > 0) {
      filter.timestamp = timestampFilter;
    }

    await dbConnect();

    if (rawUserId) {
      if (/^[0-9a-fA-F]{24}$/.test(rawUserId)) {
        filter.user_id = new Types.ObjectId(rawUserId);
      } else {
        const user = await User.findOne({
          legacy_id: rawUserId,
        })
          .select('_id')
          .lean();

        // An empty $in condition safely returns no matching logs.
        filter.user_id = user?._id ?? { $in: [] };
      }
    }

    const offset = (page - 1) * pageSize;

    const [logs, total] = await Promise.all([
      Log.find(filter)
        .sort({ [sortBy]: sortDirection })
        .skip(offset)
        .limit(pageSize)
        .populate({
          path: 'user_id',
          select: 'legacy_id',
        })
        .lean(),
      Log.countDocuments(filter),
    ]);

    const totalPages = Math.ceil(total / pageSize);

    logger.info('Admin accessed logs API', {
      source: 'api',
      url: '/api/logs',
      user_id: authUser.userId,
      filters: {
        page,
        pageSize,
        level,
        source,
        startDate,
        endDate,
        search,
        user_id: rawUserId ?? undefined,
      },
    });

    return NextResponse.json({
      success: true,
      data: logs.map(toLogDTO),
      pagination: {
        page,
        pageSize,
        total,
        totalPages,
        hasNext: page < totalPages,
        hasPrev: page > 1,
      },
    });
  } catch (error) {
    logger.error(
      `Unexpected error in logs API: ${error instanceof Error ? error.message : String(error)
      }`,
      {
        source: 'api',
        url: '/api/logs',
      },
    );

    return NextResponse.json(
      { success: false, message: 'Internal Server Error' },
      { status: 500 },
    );
  }
}

// DELETE /api/logs?olderThanDays=N
// Deletes logs older than the requested number of days.
export async function DELETE(request: NextRequest) {
  const authUser = getAuthUser(request);

  try {
    if (!authUser.isAuthenticated || !authUser.userId) {
      return errorResponse(
        'User not authenticated',
        401,
        'UNAUTHORIZED',
        request,
        authUser.userId,
      );
    }

    if (!authUser.isAdmin) {
      return errorResponse(
        'Forbidden - Admin only',
        403,
        'FORBIDDEN',
        request,
        authUser.userId,
      );
    }

    const url = new URL(request.url);
    const rawDays = Number.parseInt(
      url.searchParams.get('olderThanDays') ?? '',
      10,
    );

    if (!Number.isInteger(rawDays) || rawDays < 1) {
      return errorResponse(
        'olderThanDays must be a positive integer',
        400,
        'INVALID_OLDER_THAN_DAYS',
        request,
        authUser.userId,
      );
    }

    const cutoff = new Date(Date.now() - rawDays * 86_400_000);

    await dbConnect();

    const result = await Log.deleteMany({
      timestamp: { $lt: cutoff },
    });

    const deletedCount = result.deletedCount ?? 0;

    logger.info(
      `Log retention: purged ${deletedCount} rows older than ${rawDays} days`,
      {
        source: 'api',
        url: '/api/logs',
        user_id: authUser.userId,
      },
    );

    return NextResponse.json({
      success: true,
      message: `Deleted ${deletedCount} log entries older than ${rawDays} days`,
      deleted: deletedCount,
    });
  } catch (error) {
    logger.error(
      `Unexpected error in log purge: ${error instanceof Error ? error.message : String(error)
      }`,
      {
        source: 'api',
        url: '/api/logs',
        user_id: authUser.userId,
      },
    );

    return errorResponse(
      'Internal Server Error',
      500,
      'INTERNAL_ERROR',
      request,
      authUser.userId,
    );
  }
}