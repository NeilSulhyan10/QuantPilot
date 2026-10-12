# QuantPilot Frontend (Next.js)

A modern React frontend for the QuantPilot quantitative portfolio management system.

## Tech Stack

- **Next.js 16** (App Router)
- **React 19**
- **Tailwind CSS 3** (via PostCSS)
- **Recharts** for data visualization
- **TypeScript**

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx                 # Landing page with mode selector
│   │   ├── portfolio-builder/       # Portfolio Analytics mode (pre-computed results)
│   │   │   └── page.tsx
│   │   ├── research-lab/            # Research Lab mode (live evaluation)
│   │   │   └── page.tsx
│   │   ├── layout.tsx               # Root layout with fonts
│   │   └── globals.css              # Global styles (Tailwind + custom)
│   ├── components/
│   │   ├── ui/                      # Reusable UI components
│   │   │   ├── Card.tsx
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Select.tsx
│   │   │   ├── Table.tsx
│   │   │   ├── Badge.tsx
│   │   │   └── Section.tsx
│   │   ├── portfolio-builder/       # Portfolio Analytics specific components
│   │   └── research-lab/            # Research Lab specific components
│   └── lib/
│       ├── api.ts                   # API client with typed endpoints
│       ├── types.ts                 # TypeScript interfaces matching Python dataclasses
│       ├── formatters.ts            # Formatting utilities (percent, currency, etc.)
│       └── utils.ts                 # clsx utility
├── public/                          # Static assets
├── tailwind.config.js               # Tailwind configuration
├── postcss.config.js                # PostCSS configuration
├── next.config.ts                   # Next.js configuration
├── tsconfig.json                    # TypeScript configuration
└── package.json                     # Dependencies and scripts
```

## Getting Started

### Prerequisites

- Node.js 18+
- Python backend running on `http://localhost:8000` (see backend README)

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### Production Build

```bash
npm run build
npm start
```

## Environment Variables

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Features

### Portfolio Analytics Mode
- Displays pre-computed backtest results from CSV files
- Performance metrics (CAGR, Sharpe, Volatility, Max Drawdown, Cumulative Return)
- Performance comparison table across strategies
- Cumulative growth and drawdown charts
- Portfolio allocation (latest & average weights)
- Portfolio characteristics (active positions, effective stocks, max position)
- Turnover chart
- Methodology and limitations documentation

### Research Lab Mode
- Market selection (US/India)
- Investment plan configuration (Lump Sum / Monthly SIP, horizon)
- Risk profile selection (Conservative/Moderate/Aggressive)
- Estimation window configuration
- Company selection with ticker normalization
- Ranking method selection
- Live subset evaluation with all 3 risk profiles
- Risk profile comparison tables
- Allocation charts and capital allocation tables
- Stock trend analysis (bullish/neutral/bearish with technical indicators)
- Historical backtest comparison (QuantPilot vs Equal Weight vs Benchmark)
- Growth and drawdown comparison charts

## API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check |
| `/api/markets` | GET | Available market configurations |
| `/api/parse-horizon` | GET | Parse investment horizon string |
| `/api/evaluate` | POST | Run subset evaluation |
| `/api/backtest` | POST | Run portfolio backtest |
| `/api/trend` | POST | Analyze stock trends |
| `/api/portfolio-builder/*` | GET | Pre-computed Portfolio Analytics data |

## Styling

The frontend replicates the Streamlit dashboard's visual design:
- Custom color palette (blue primary, slate grays, semantic colors)
- Space Grotesk for headings, DM Sans for body
- Card-based layout with subtle shadows
- Consistent spacing and typography
- Responsive design (mobile-first)

## TypeScript Types

All API responses are fully typed in `src/lib/types.ts`, matching the Python dataclasses:
- `MarketConfig`
- `SubsetInvestmentResult`
- `RiskProfileResult`
- `StockAnalysis`
- `BacktestResponse`
- `PortfolioBuilderData`
- And more...

## Backend Integration

The frontend communicates with the Python FastAPI backend (`main.py`) running on port 8000. Ensure the backend is running before starting the frontend.

```bash
# Terminal 1: Start backend
cd ..
python main.py

# Terminal 2: Start frontend
cd frontend
npm run dev
```

## Removed Dependencies

- Streamlit (replaced by Next.js)
- All Streamlit-specific code removed from requirements.txt