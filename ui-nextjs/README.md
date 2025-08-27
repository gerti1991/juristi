# 🌐 Juristi AI - Next.js Frontend

Modern, responsive web interface for the Albanian Legal AI Assistant built with Next.js 15, React 19, and TypeScript.

## 🚀 Features

- **Modern UI/UX**: Grok AI-inspired design with professional legal styling
- **Dual Query Modes**: Switch between Precise and Analyzed legal research
- **Real-time Processing**: Live query status with loading states  
- **Source Verification**: Detailed source display with metadata
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **TypeScript**: Full type safety throughout the application
- **Fast Performance**: Next.js 15 with React 19 optimizations

## 🛠️ Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Next.js** | 15.5.2 | React framework with App Router |
| **React** | 19.1.1 | UI library with concurrent features |
| **TypeScript** | 5.9.2 | Static type checking |
| **Tailwind CSS** | 3.4.1 | Utility-first CSS framework |
| **Lucide React** | 0.542.0 | Modern icon library |
| **React Markdown** | 9.0.1 | Markdown rendering with GFM support |

## 📦 Installation

```bash
# Navigate to the frontend directory
cd ui-nextjs

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the `ui-nextjs` directory:

```bash
# FastAPI Backend URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional: Custom configuration
NEXT_PUBLIC_APP_NAME="Juristi AI"
```

### Next.js Configuration

The `next.config.js` includes optimizations for:
- Image optimization with external domains
- Development environment origins
- Build-time optimizations

## 📁 Project Structure

```
ui-nextjs/
├── src/
│   ├── app/
│   │   ├── globals.css          # Global styles
│   │   ├── layout.tsx           # Root layout component
│   │   └── page.tsx             # Home page
│   ├── components/
│   │   ├── SearchInput.tsx      # Search input component
│   │   └── SearchInterface.tsx  # Main search interface
│   ├── lib/
│   │   └── types.ts            # TypeScript type definitions
│   └── types/
│       └── api.ts              # API response types
├── public/                     # Static assets
├── package.json               # Dependencies and scripts
├── tailwind.config.js         # Tailwind CSS configuration
├── tsconfig.json             # TypeScript configuration
└── next.config.js            # Next.js configuration
```

## 🎯 Component Architecture

### SearchInterface (Main Component)
- **Purpose**: Primary interface handling dual-mode queries
- **State Management**: React hooks for query state and responses
- **API Integration**: Direct fetch calls to FastAPI backend
- **Error Handling**: Comprehensive error states and user feedback

### SearchInput
- **Purpose**: Query input with mode selection
- **Features**: Mode switching, loading states, form validation
- **Accessibility**: Full keyboard navigation and screen reader support

## 🔌 API Integration

The frontend communicates directly with the FastAPI backend:

```typescript
// Search request
const response = await fetch(`${API_URL}/search`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: userQuery,
    mode: selectedMode // 'precise' | 'analyzed'
  }),
});
```

## 🎨 Styling & Theming

### Grok AI-Inspired Design
- **Color Scheme**: Professional legal blue with accent colors
- **Typography**: Clean, readable fonts optimized for legal text
- **Layout**: Modern card-based design with proper spacing
- **Responsive**: Mobile-first approach with breakpoint optimizations

### Mode-Specific Styling
- **Precise Mode**: Green accents for focused queries
- **Analyzed Mode**: Purple accents for comprehensive analysis
- **Loading States**: Smooth animations and progress indicators

## 🚀 Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Production deployment
vercel --prod
```

### Docker
```dockerfile
FROM node:18-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS builder
WORKDIR /app
COPY . .
COPY --from=deps /app/node_modules ./node_modules
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
```

### Manual Build
```bash
# Build the application
npm run build

# Start production server
npm start
```

## 🧪 Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build production application
- `npm start` - Start production server
- `npm run lint` - Run ESLint for code quality

### Development Guidelines

1. **Component Structure**: Use functional components with TypeScript
2. **State Management**: Prefer React hooks over external state libraries
3. **API Calls**: Use native fetch with proper error handling
4. **Styling**: Tailwind CSS with component-specific styles
5. **Type Safety**: Full TypeScript coverage with strict mode

### Testing

```bash
# Install testing dependencies (when added)
npm install --save-dev @testing-library/react @testing-library/jest-dom jest

# Run tests (when implemented)
npm test
```

## 📱 Browser Support

- **Chrome**: 88+
- **Firefox**: 78+
- **Safari**: 14+
- **Edge**: 88+

## 🤝 Contributing

1. Follow the existing code style and structure
2. Ensure TypeScript types are properly defined
3. Test on multiple screen sizes and browsers
4. Update documentation for new features

## 📄 License

This project is part of the Juristi AI system and follows the same license terms.

---

**Built with ❤️ using Next.js 15, React 19, and TypeScript**
