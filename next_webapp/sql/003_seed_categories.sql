-- Insert sample categories
INSERT INTO categories (name, description, slug, color, icon) VALUES
('Technology', 'Technology and software related content', 'technology', '#3B82F6', 'tech'),
('Environment', 'Environmental sustainability and green initiatives', 'environment', '#10B981', 'leaf'),
('Transport', 'Transportation and mobility solutions', 'transport', '#F59E0B', 'car'),
('Health', 'Healthcare and wellbeing', 'health', '#EF4444', 'heart'),
('Education', 'Educational resources and learning', 'education', '#8B5CF6', 'book'),
('Data Science', 'Data analysis and machine learning', 'data-science', '#6366F1', 'chart'),
('Urban Planning', 'City development and planning', 'urban-planning', '#EC4899', 'building'),
('Safety', 'Public safety and security', 'safety', '#F97316', 'shield')
ON CONFLICT (slug) DO NOTHING;
