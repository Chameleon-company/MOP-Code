// database.tsx
export type CaseStudy = {
  id: number;
  title: string;
  content: string;
  group: string;
  pdf?: string;          // Optional PDF link
  caseUrl?: string;      // Optional API endpoint or URL for more dynamic content retrieval
};

export const database: CaseStudy[] = [
  { id: 1, title: 'Case Study 1', content: 'Content for Case Study 1...', group: 'Internet', pdf: '/pdf/one.pdf', caseUrl: "/api?filename=Childcare_Facilities_Analysis" },
  { id: 2, title: 'Case Study 2', content: 'Content for Case Study 2...', group: 'Internet', pdf: '/MOP-Code/next_webapp/src/app/UseCases/Task.pdf', caseUrl: "/api?filename=Projected_Music_venue_growth" },
  { id: 3, title: 'Case Study 3', content: 'Content for Case Study 3...', group: 'Internet' },  // Example without PDF or caseUrl
  { id: 4, title: 'Case Study 4', content: 'Content for Case Study 4...', group: 'Internet' },  // Example without PDF or caseUrl
  { id: 5, title: 'Case Study 5', content: 'Content for Case Study 5...', group: 'Internet' },  // Continues similarly
  { id: 6, title: 'Case Study 6', content: 'Content for Case Study 6...', group: 'EV', caseUrl: "/api?filename=EV_Charging_Analysis" },
  { id: 7, title: 'Case Study 7', content: 'Content for Case Study 7...', group: 'EV', caseUrl: "/api?filename=EV_Market_Study" },
  { id: 8, title: 'Case Study 8', content: 'Content for Case Study 8...', group: 'EV' },
  { id: 9, title: 'Case Study 9', content: 'Content for Case Study 9...' , group: 'EV'},
    { id: 10, title: 'Case Study 10', content: 'Content for Case Study 10...', group: 'EV' },
    { id: 11, title: 'Case Study 11', content: 'Content for Case Study 11...' , group: 'Security'},
    { id: 12, title: 'Case Study 12', content: 'Content for Case Study 12...' , group: 'Security'},
    { id: 13, title: 'Case Study 13', content: 'Content for Case Study 13...', group: 'Security' },
    { id: 14, title: 'Case Study 14', content: 'Content for Case Study 14...' , group: 'Security'},
    { id: 15, title: 'Case Study 15', content: 'Content for Case Study 15...' , group: 'Security'},
    { id: 16, title: 'Case Study 16', content: 'Content for Case Study 16...', group: 'Security' },
    { id: 17, title: 'Case Study 17', content: 'Content for Case Study 17...' , group: 'Security'}
    // Add additional case studies as needed
];
