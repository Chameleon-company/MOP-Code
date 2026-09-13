/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import AddCategoryPage from '../../app/[locale]/admin/categories/add/page';
import EditCategoryPage from '../../app/[locale]/admin/categories/edit/[id]/page';
import CategoriesPage from '../../app/[locale]/admin/categories/page';

const mockPush = jest.fn();
const mockReplace = jest.fn();
const mockParams = { locale: 'en', id: '42' };

jest.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush, replace: mockReplace }),
  useParams: () => mockParams,
}));

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ href, children, ...props }: any) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

jest.mock('@/components/admin/AdminToast', () => ({
  __esModule: true,
  default: ({ message, type }: { message: string; type: string }) => (
    <div data-testid="toast">{type}: {message}</div>
  ),
}));

jest.mock('@/components/admin/ConfirmModal', () => ({
  __esModule: true,
  default: ({ open, title, message, onConfirm, onCancel }: any) =>
    open ? (
      <div>
        <h2>{title}</h2>
        <p>{message}</p>
        <button onClick={onConfirm}>Delete</button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    ) : null,
}));

jest.mock('@/components/admin/ImageHoverPreview', () => ({
  __esModule: true,
  default: ({ src, alt }: any) => <img src={src} alt={alt} data-testid="image-preview" />,
}));

jest.mock('@/components/admin/TextHoverPreview', () => ({
  __esModule: true,
  default: ({ text }: any) => <span>{text}</span>,
}));

jest.mock('@/components/Pagination', () => ({
  __esModule: true,
  default: ({ page, totalPages, onPageChange }: any) => (
    <div>
      <span>Page {page} of {totalPages}</span>
      <button onClick={() => onPageChange(1)}>Set page 1</button>
    </div>
  ),
}));

describe('Admin category CRUD flows', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    Object.defineProperty(URL, 'createObjectURL', {
      writable: true,
      value: jest.fn(() => 'blob:mock-url'),
    });
    Object.defineProperty(URL, 'revokeObjectURL', {
      writable: true,
      value: jest.fn(),
    });
    mockPush.mockClear();
    mockReplace.mockClear();
    window.localStorage.clear();
    window.localStorage.setItem(
      'user',
      JSON.stringify({
        userId: 'u-1',
        roleId: 'admin',
        token: 'secret-token',
        roleName: 'admin',
      }),
    );
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
    jest.resetAllMocks();
  });

  test('creates a category with image upload and navigates back to the list', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const mockFetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, url: 'https://cdn.example.com/cats/urban.jpg' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });
    global.fetch = mockFetch as jest.Mock;

    render(<AddCategoryPage />);

    await user.type(screen.getByPlaceholderText('Enter category title'), 'Urban Mobility');
    await user.type(screen.getByPlaceholderText('Enter category description'), 'City transport and access');

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['image'], 'urban.png', { type: 'image/png' });
    await user.upload(fileInput, file);

    await user.click(screen.getByRole('button', { name: /save category/i }));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(2));

    expect(mockFetch).toHaveBeenNthCalledWith(
      1,
      '/api/upload',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'x-user-id': 'u-1',
          'x-user-role-id': 'admin',
        }),
      }),
    );

    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      '/api/categories',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          category_name: 'Urban Mobility',
          description: 'City transport and access',
          cover_img: 'https://cdn.example.com/cats/urban.jpg',
        }),
      }),
    );

    await waitFor(() => {
      expect(screen.getByText(/Category added successfully\./i)).toBeInTheDocument();
    });

    jest.advanceTimersByTime(1000);
    expect(mockPush).toHaveBeenCalledWith('/en/admin/categories');
  });

  test('loads category values and updates them on save', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const mockFetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: {
            category_name: 'Old Category',
            description: 'Old description',
            cover_img: 'https://cdn.example.com/old.png',
          },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });
    global.fetch = mockFetch as jest.Mock;

    render(<EditCategoryPage />);

    await waitFor(() => {
      expect(screen.getByDisplayValue('Old Category')).toBeInTheDocument();
    });

    const titleInput = screen.getByPlaceholderText('Enter category title');
    const descriptionInput = screen.getByPlaceholderText('Enter category description');
    await user.clear(titleInput);
    await user.type(titleInput, 'Updated Category');
    await user.clear(descriptionInput);
    await user.type(descriptionInput, 'Updated description');
    await user.click(screen.getByRole('button', { name: /update category/i }));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(2));

    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      '/api/categories/42',
      expect.objectContaining({
        method: 'PUT',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          category_name: 'Updated Category',
          description: 'Updated description',
          cover_img: 'https://cdn.example.com/old.png',
        }),
      }),
    );

    expect(mockPush).toHaveBeenCalledWith('/en/admin/categories');
  });

  test('lists categories and confirms delete flow', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const mockFetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: [{ id: 1, category_name: 'Transport', description: 'Mobility', cover_img: null }],
          pagination: { totalPages: 1, total: 1 },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });
    global.fetch = mockFetch as jest.Mock;

    render(<CategoriesPage />);

    await waitFor(() => {
      expect(screen.getByText('Transport')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /delete transport/i }));

    await waitFor(() => {
      expect(screen.getByText(/Are you sure you want to delete/i)).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /^delete$/i }));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(3));
    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      '/api/categories/1',
      expect.objectContaining({ method: 'DELETE' }),
    );

    await waitFor(() => {
      expect(screen.getByText(/Category deleted successfully\./i)).toBeInTheDocument();
    });
  });
});
