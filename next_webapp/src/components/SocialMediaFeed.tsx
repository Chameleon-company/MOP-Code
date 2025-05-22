'use client';
import React from 'react';
import { LinkedInEmbed } from 'react-social-media-embed';

const SocialMediaFeed: React.FC = () => {
  return (
    <div className="w-full bg-white dark:bg-[#263238] py-8 text-black dark:text-white">
      <section className="recent-case-studies text-center">
        <h2 className="text-2xl font-bold mb-4">Connect with us on LinkedIn</h2>
      </section>

      <div className="flex justify-center px-4">
        <div className="w-full max-w-[600px]">
          <LinkedInEmbed
            url="https://www.linkedin.com/company/chameleon-smarter-world/"
            width="100%"
          />
        </div>
      </div>
    </div>
  );
};

export default SocialMediaFeed;
