'use client';
import React from 'react';
import { InstagramEmbed } from 'react-social-media-embed';

const SocialMediaFeed: React.FC = () => {
  return (
    <section
      className="w-full bg-white dark:bg-[#263238] py-8 px-4 text-black dark:text-white"
      aria-labelledby="instagram-heading"
    >
      <div className="text-center mb-6">
        <h2 id="instagram-heading" className="text-2xl md:text-3xl font-bold">
          Follow us on Instagram
        </h2>
        <p className="mt-2 text-sm md:text-base text-gray-600 dark:text-gray-300">
          Explore the latest updates, events, and city stories.
        </p>
      </div>

      <div className="flex justify-center">
        <div className="w-full max-w-[600px]">
          <InstagramEmbed
            url="https://www.instagram.com/cityofmelbourne/?igsh=OXFqa25sdno5OTRv#"
            width="100%"
          />
        </div>
      </div>
    </section>
  );
};

export default SocialMediaFeed;
