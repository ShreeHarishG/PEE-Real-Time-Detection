import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
      {
        source: '/results/:path*',
        destination: 'http://127.0.0.1:8000/results/:path*',
      },
      {
        source: '/evidence/:path*',
        destination: 'http://127.0.0.1:8000/evidence/:path*',
      },
      {
        source: '/uploads/:path*',
        destination: 'http://127.0.0.1:8000/uploads/:path*',
      }
    ];
  },
};

export default nextConfig;
