describe('MongoDB API smoke tests', () => {
  it('returns paginated use cases from MongoDB', () => {
    cy.request('/api/usecases?page=1&pageSize=5').then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body.success).to.eq(true);
      expect(response.body.data).to.be.an('array');
      expect(response.body.count).to.eq(response.body.data.length);

      expect(response.body.pagination).to.include({
        page: 1,
        pageSize: 5,
      });

      expect(response.body.pagination.total).to.be.a('number');
      expect(response.body.pagination.totalPages).to.be.a('number');
    });
  });

  it('returns the MongoDB search response expected by the frontend', () => {
    cy.request('/api/search?page=1&pageSize=5').then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body.success).to.eq(true);
      expect(response.body.data.results).to.be.an('array');

      expect(response.body.data.pagination).to.include({
        page: 1,
        pageSize: 5,
      });

      expect(response.body.data.pagination.total).to.be.a('number');
      expect(response.body.data.filters).to.be.an('object');
    });
  });

  it('returns the total use-case count from MongoDB', () => {
    cy.request('/api/statistics/total-count').then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body.success).to.eq(true);
      expect(response.body.total).to.be.a('number');
    });
  });

  it('rejects an unauthenticated use-case creation request', () => {
    cy.request({
      method: 'POST',
      url: '/api/usecases',
      failOnStatusCode: false,
      body: {
        title: 'Cypress should not create this',
      },
    }).then((response) => {
      expect(response.status).to.eq(401);
      expect(response.body.success).to.eq(false);
    });
  });

  it('rejects unauthenticated access to logs', () => {
    cy.request({
      url: '/api/logs',
      failOnStatusCode: false,
    }).then((response) => {
      expect(response.status).to.eq(401);
    });
  });
});