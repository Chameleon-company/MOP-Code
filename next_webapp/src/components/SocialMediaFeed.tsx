'use client';
import React from 'react';
import {
  LinkedInEmbed,
} from 'react-social-media-embed';

const SocialMediaFeed: React.FC = () => {
  return (
    <div className="w-full bg-white py-8">
        <section className="recent-case-studies">
            <h2>{("Connect with us on Linkedin")}</h2>
            </section>
        {/* LinkedIn */}
        <div className="flex flex-col items-center">
          <LinkedInEmbed
            url="https://www.linkedin.com/company/chameleon-smarter-world/"
            width={600}
          />
        </div>
      </div>
  );
};

export default SocialMediaFeed;
