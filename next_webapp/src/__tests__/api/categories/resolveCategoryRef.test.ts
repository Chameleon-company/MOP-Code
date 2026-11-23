/**
 * @jest-environment node
 *
 * Tests for _shared.js' resolveCategoryRef()
 *
 * All Mongoose calls are mocked — no real DB work happens here.
 */

import getGridFSBucket from "@/lib/gridfs";

// ─── Mocks ──────────────────────────────────────────────────────────────────

jest.mock('../../../models/mongoose/Category', () => ({
    __esModule: true,
    default: {findById: jest.fn(), findOne: jest.fn()}
}));

jest.mock('../../../models/mongoose/Tag', () => ({
    __esModule: true,
    default: {}
}));

jest.mock('../../../models/mongoose/User', () => ({
    __esModule: true,
    default: {}
}));

jest.mock('../../../lib/gridfs', () => ({
    getGridFSBucket: jest.fn()
}));

// ─── Imports ────────────────────────────────────────────────────────────────

import mongoose from 'mongoose';
import Category from '../../../models/mongoose/Category';
import {resolveCategoryRef, InvalidCategoryRefError} from '../../../app/api/usecases/_shared';

const jestExpect = globalThis.expect as unknown as jest.Expect;

// ─── Helpers ────────────────────────────────────────────────────────────────

const leanResolves = (value: unknown) => ({lean: () => Promise.resolve(value)});

// ─── Tests ──────────────────────────────────────────────────────────────────

beforeEach(() => {
    jest.clearAllMocks();
    (Category.findById as jest.Mock).mockReturnValue(leanResolves(null));
    (Category.findOne as jest.Mock).mockReturnValue(leanResolves(null));
});

describe('resolveCategoryRef() - invalid input', () => {
    const INVALID_INPUTS: unknown[] = [
        'NaN',
        'undefined',
        'null',
        ' ',
        ' 13 ',
        '5.5',
        '-4',
        'asdfghjkl',
        ['12'],
        'Infinity',
        '{}'
    ];

    test.each(INVALID_INPUTS.map((v) => [v]))(
        'rejects %p without querying MongoDB',
        async(input) => {
            await expect(resolveCategoryRef(input as any)).rejects.toThrow(
                InvalidCategoryRefError
            );
            expect(Category.findById).not.toHaveBeenCalled();
            expect(Category.findOne).not.toHaveBeenCalled();
        }
    )
});

describe('resolveCategoryRef - inputs that must keep working', () => {
    test.each([[undefined], [null], ['']])('returns as null for %p', async(input) => {
        await expect(resolveCategoryRef(input as any)).resolves.toBeNull();
    });

    test('resolve a real ObjectId', async() => {
        const _id = new mongoose.Types.ObjectId();
        (Category.findById as jest.Mock).mockReturnValue(
            leanResolves({_id, legacy_id: '6', category_name: 'Cars'})
        );
        await expect(resolveCategoryRef(String(_id))).resolves.toEqual({
            id: _id, legacy_id: '6', category_name: 'Cars'
        });
    });

    test('resolve a numeric legacy id as string or number', async() => {
        const _id = new mongoose.Types.ObjectId();
        (Category.findOne as jest.Mock).mockReturnValue(
            leanResolves({_id, legacy_id: '15', category_name: 'Trains'})
        );
        expect((await resolveCategoryRef('15'))?.legacy_id).toBe('15');
        expect((await resolveCategoryRef(15 as any))?.legacy_id).toBe('15');
    })

    test('keep unresolved legacy fallback for unknown numeric id', async() => {
        await expect(resolveCategoryRef('777777')).resolves.toEqual({
            id: null, legacy_id: '777777', category_name: null
        });
    });

    test('well-formed ObjectId with no match - falls back', async() =>{
        const testId = String(new mongoose.Types.ObjectId());
        await expect(resolveCategoryRef(testId)).resolves.toEqual({
            id: null, legacy_id: testId, category_name: null
        });
    });
});