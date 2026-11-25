describe("Public Use Case Explorer category filter", () => {
  const transport = { id: "64a000000000000000000001", category_name: "Transport" };
  const environment = { id: "64a000000000000000000002", category_name: "Environment" };
  const categoryControl = "#use-case-category";
  const searchInput = 'input[placeholder="Search use cases"]';

  beforeEach(() => {
    cy.intercept({ method: "GET", pathname: "/api/usecases" }, (request) => {
      const { category_id, search, page } = request.query;
      const title = category_id === environment.id
        ? null
        : category_id === transport.id
          ? page === "2" ? "Tram study page two" : "Tram study"
          : "City study";
      const totalPages = title ? 2 : 0;
      request.reply({
        success: true,
        data: title ? [{ id: "study-1", title, description: "A study", tags: [] }] : [],
        pagination: { total: title ? 10 : 0, totalPages },
      });

      if (category_id === transport.id && search === "tram") request.alias = "combined";
      else if (category_id === transport.id) request.alias = "transport";
      else if (category_id === environment.id) request.alias = "environment";
      else if (search) request.alias = "allWithSearch";
      else request.alias = "all";
    });
  });

  it("loads API categories, combines filters, resets pagination and clears the category", () => {
    cy.intercept("GET", "/api/home/categories", {
      body: { success: true, data: [transport, environment] },
      delay: 800,
    }).as("categories");

    cy.visit("./usecases");
    cy.contains("label", "Category").should("have.attr", "for", "use-case-category");
    cy.get(categoryControl).should("have.value", "");
    cy.wait("@categories");
    cy.get(categoryControl).find("option").should("have.length", 3);
    cy.get(categoryControl).select("Transport");
    cy.wait("@transport").its("request.query").should("deep.include", {
      category_id: transport.id, page: "1", pageSize: "9",
    });
    cy.contains("Tram study").should("be.visible");
    cy.contains("City study").should("not.exist");

    cy.get(searchInput).type("tram");
    cy.wait("@combined").its("request.query").should("deep.include", {
      category_id: transport.id, search: "tram", search_by: "title", page: "1", pageSize: "9",
    });
    cy.contains("span", "Page 1 of 2").next("button").click();
    cy.wait("@combined").its("request.query").should("deep.include", {
      category_id: transport.id, search: "tram", search_by: "title", page: "2", pageSize: "9",
    });
    cy.contains("Tram study page two").should("be.visible");

    cy.get(categoryControl).select("Environment");
    cy.wait("@environment").its("request.query").should("deep.include", {
      category_id: environment.id, search: "tram", search_by: "title", page: "1", pageSize: "9",
    });
    cy.contains("No use cases match these filters. Try a different search or select All categories.")
      .should("be.visible");

    cy.get(categoryControl).select("All categories");
    cy.wait("@allWithSearch").its("request.query").should((query) => {
      expect(query).to.deep.include({ search: "tram", search_by: "title", page: "1", pageSize: "9" });
      expect(query).not.to.have.property("category_id");
    });
    cy.get(searchInput).should("have.value", "tram");
    cy.contains("City study").should("be.visible");
  });

  it("preserves tag search when changing categories", () => {
    cy.intercept("GET", "/api/home/categories", { success: true, data: [transport] });
    cy.visit("./usecases?tag_name=transit");
    cy.get(categoryControl).select("Transport");
    cy.wait("@transport").its("request.query").should((query) => {
      expect(query).to.deep.include({
        category_id: transport.id, tag_name: "transit", page: "1", pageSize: "9",
      });
      expect(query).not.to.have.property("search");
    });
  });

  it("keeps search usable when categories fail to load", () => {
    cy.intercept("GET", "/api/home/categories", { statusCode: 500, body: { success: false } });
    cy.visit("./usecases");
    cy.contains("Could not load categories. You can still search.").should("be.visible");
    cy.get(categoryControl).should("be.disabled");
    cy.get(searchInput).type("tram");
    cy.wait("@allWithSearch");
    cy.contains("City study").should("be.visible");
  });

  it("shows feedback when there are no categories", () => {
    cy.intercept("GET", "/api/home/categories", { success: true, data: [] });
    cy.visit("./usecases");
    cy.contains("No categories available.").should("be.visible");
    cy.get(categoryControl).should("be.disabled").and("have.value", "");
    cy.contains("City study").should("be.visible");
  });
});
