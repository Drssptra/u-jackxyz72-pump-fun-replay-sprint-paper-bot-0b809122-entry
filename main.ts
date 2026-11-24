// Pump.fun Replay Sprint — deterministic paper trading bot.
//
// Strategy (designed from scratch for this competition, from the dev
// dataset's shape, not from any baseline):
//
// Pump.fun coins spike fast and fade fast: in the dev window the median
// coin hits its peak only ~8 slots after its first trade, and the median
// coin ends the window below its first price. Two facts from the dev
// data drive the design:
//
//  1. Momentum persists: coins that already ran up hard by the time they
//     cross the entry threshold go on to peak much higher than coins
//     that barely moved (a coin up 2x at decision peaks a further 1.26x
//     at the median; one up barely 1.0-1.05x peaks only 1.15x). So the
//     bot buys the strong pumpers, not the quiet ones — and never caps
//     how far a coin may already have run.
//  2. The pump happens fast and mostly inside single slots, so the bot
//     enters as soon as the size cap lets it buy a full 0.01 SOL stake
//     (real reserve 1 SOL) — any later and the pump is already over —
//     but only for coins that already ran 1.4x from their first trade
//     and whose latest trade was a buy.
//
//  - Entry: the first batch where a new curve's real SOL reserve reaches
//    1 SOL (so the 1%-of-real-reserve size cap lets us buy the 0.01 SOL
//    minimum round trip), its price is at least 1.4x its first trade,
//    and its latest trade in the batch was a buy. The decision is made
//    once, on that batch only: a coin whose slot is taken when it
//    crosses is dropped for good, never chased later at a pumped price.
//    One buy per mint. At most 5 open positions, at most 16 orders per
//    batch.
//  - Exit: sell the whole position on the trailing stop (armed at +15%,
//    fires 5% below the peak since entry), on the +100% take-profit, at
//    the -12% stop loss, or after 300 slots held (~2 minutes) without an
//    exit firing. Peaks are transient (the median peak fades 10% within
//    ~18 slots), so the tight trailing stop is what banks a spike;
//    the fixed take-profit is only a safety valve for a moonshot.
//  - Sells use wide slippage so an exit always executes; buys use 5%
//    slippage so a fill two slots later never chases a spike.
//  - Every round trip costs ~2.5% in curve fees plus two 105,000-lamport
//    transaction fees, so a 0.01 SOL stake needs roughly a 4.6% move to
//    break even; the trailing exit banks spikes well above that.
//
// Determinism: every decision comes only from the event stream the
// harness delivers (slots, reserves, fills). No clock, no environment,
// no Math.random. Running the bot twice over the same window produces
// the same action log byte for byte.
//
// Guardrails: 0.01 SOL stakes clear the 0.01 SOL minimum round trip,
// the bot cycles through far more than 20 distinct mints, and with at
// most 0.05 SOL deployed at once the drawdown stays far under 50%.

import { pathToFileURL } from 'node:url';
import {
  runBot,
  type Batch,
  type BotState,
  type OrderInput,
} from './vendor/makefun-bot.ts';

/** Lamports in one SOL. */
const LAMPORTS_PER_SOL = 1_000_000_000n;
/** Minimum buy for a round trip to count toward the guardrail. */
const MIN_ROUND_TRIP_LAMPORTS = 10_000_000n;

export interface BotParams {
  /** SOL spent per entry (before fees). */
  stakeLamports: bigint;
  /** Enter once a curve's real SOL reserve reaches this. */
  minRealReserveLamports: bigint;
  /** Enter only when the price is at least this multiple of its first trade. */
  minEntryMultipleBps: number;
  /** Enter only when the mint's last trade in the batch was a buy. */
  requireLastTradeBuy: boolean;
  /** Never enter when the price already ran past this multiple of its first trade. */
  maxEntryMultipleBps: number;
  /** Buy fill rejected when it comes back worse than this. */
  buySlippageBps: number;
  /** Sells always execute, even into a dump. */
  sellSlippageBps: number;
  /** Sell when mark price is this far above cost per token. */
  takeProfitBps: number;
  /** Sell when mark price is this far below cost per token. */
  stopLossBps: number;
  /** Sell a position held longer than this many slots. */
  maxHoldSlots: number;
  /** Arm the trailing exit once the mark is this far above cost per token. */
  trailArmBps: number;
  /** Once armed, sell when the mark falls this far below its peak since entry. */
  trailDropBps: number;
}

export const DEFAULT_PARAMS: BotParams = {
  stakeLamports: (LAMPORTS_PER_SOL * 1n) / 100n, // 0.01 SOL
  minRealReserveLamports: LAMPORTS_PER_SOL * 1n, // 1 SOL of real reserve
  minEntryMultipleBps: 14_000, // at least 1.4x its first trade price
  requireLastTradeBuy: true,
  maxEntryMultipleBps: 1_000_000, // no effective ceiling (momentum persists)
  buySlippageBps: 500, // 5%
  sellSlippageBps: 10_000, // no effective limit
  takeProfitBps: 10_000, // +100% (safety valve; the trailing exit banks spikes)
  stopLossBps: 1200, // -12%
  maxHoldSlots: 300,
  trailArmBps: 1500, // arm the trailing exit at +15%
  trailDropBps: 500, // sell 5% below the peak since entry
};

interface MintNote {
  /** Price (lamports per token) at the mint's first trade. */
  firstPrice: number;
  /** True once the bot has decided on this mint for good. */
  decided: boolean;
}

export function createBot(params: BotParams = DEFAULT_PARAMS) {
  const notes = new Map<string, MintNote>();
  /** Slot a buy fill landed, per mint still held. */
  const entrySlot = new Map<string, number>();
  /** Highest mark/cost ratio seen since entry, per mint still held. */
  const peakRatio = new Map<string, number>();
  /** Mints with a sell already in flight. */
  const pendingSell = new Set<string>();

  /** Current price of a curve in lamports per token, from virtual reserves. */
  function priceOf(curve: {
    virtual_token_reserves: bigint;
    virtual_quote_reserves: bigint;
  }): number | null {
    if (curve.virtual_token_reserves <= 0n || curve.virtual_quote_reserves <= 0n) {
      return null;
    }
    return Number(curve.virtual_quote_reserves) / Number(curve.virtual_token_reserves);
  }

  function onBatch(state: BotState, batch: Batch): OrderInput[] {
    // Fold the fills that resolved since the last batch: a buy fill starts
    // the holding clock, a sell fill (or a rejected sell) frees the mint.
    for (const fill of batch.fills) {
      if (fill.side === 'buy' && fill.status === 'filled') {
        entrySlot.set(fill.mint, fill.slot);
        peakRatio.set(fill.mint, 1);
      } else if (fill.side === 'sell') {
        pendingSell.delete(fill.mint);
        if (fill.status === 'filled') {
          entrySlot.delete(fill.mint);
          peakRatio.delete(fill.mint);
        }
      }
    }

    // Note every curve the first time it trades, at its first trade price
    // (the create price is the same constant for every coin, so the first
    // trade is the meaningful baseline for "how far has this run").
    for (const event of batch.events) {
      if (event.event !== 'TradeEvent' || notes.has(event.mint)) continue;
      const price = priceOf({
        virtual_token_reserves: BigInt(event.virtual_token_reserves),
        virtual_quote_reserves: BigInt(event.virtual_quote_reserves),
      });
      if (price !== null) notes.set(event.mint, { firstPrice: price, decided: false });
    }

    const orders: OrderInput[] = [];
    const tx = state.txFeeLamports();

    // Exits first: take profit, stop loss, or time exit on every holding.
    for (const [mint, holding] of state.positions) {
      if (pendingSell.has(mint)) continue;
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      const mark = priceOf(curve);
      if (mark === null) continue;
      const entryPrice = Number(holding.cost_lamports) / Number(holding.token_amount);
      if (!(entryPrice > 0)) continue;
      const ratio = mark / entryPrice;
      const heldSlots = state.slot - (entrySlot.get(mint) ?? state.slot);
      const peak = Math.max(peakRatio.get(mint) ?? 1, ratio);
      peakRatio.set(mint, peak);
      const takeProfitAt = 1 + params.takeProfitBps / 10_000;
      const stopLossAt = 1 - params.stopLossBps / 10_000;
      // Trailing exit: once the position has run up to the arm level, sell
      // when it gives back the trailing drop from its peak since entry.
      const trailing =
        peak >= 1 + params.trailArmBps / 10_000 &&
        ratio <= peak * (1 - params.trailDropBps / 10_000);
      if (ratio >= takeProfitAt || ratio <= stopLossAt || heldSlots >= params.maxHoldSlots || trailing) {
        orders.push({
          side: 'sell',
          mint,
          token_amount: holding.token_amount,
          max_slippage_bps: params.sellSlippageBps,
        });
        pendingSell.add(mint);
      }
    }

    // The direction of each mint's last trade in this batch: a coin whose
    // latest trade was a sell is already fading.
    const lastTradeBuy = new Map<string, boolean>();
    for (const event of batch.events) {
      if (event.event === 'TradeEvent') lastTradeBuy.set(event.mint, event.is_buy);
    }

    // Entries: new curves whose real reserve reached the threshold, in
    // event order. A coin is decided exactly once, on the first batch
    // where it meets the entry criteria: if no slot, cash, or order room
    // is free then, it is dropped for good rather than chased later at a
    // pumped price.
    let free = state.freeCashLamports();
    let slots = state.hello.caps.max_open_positions - state.committedPositionCount();
    for (const event of batch.events) {
      const note = notes.get(event.mint);
      if (note === undefined || note.decided) continue;
      const curve = state.curves.get(event.mint);
      if (curve === undefined || curve.complete) {
        note.decided = true;
        continue;
      }
      // Not enough real reserve yet: wait for a later batch to decide.
      if (curve.real_quote_reserves < params.minRealReserveLamports) continue;
      if (curve.trades < 1) continue;
      note.decided = true;
      const now = priceOf(curve);
      if (now === null) continue;
      // Momentum confirmation: only buy a coin that already ran up from
      // its first trade (strong early pumpers peak higher afterwards).
      if (now < (note.firstPrice * params.minEntryMultipleBps) / 10_000) continue;
      // And whose latest trade in this batch was a buy, not a sell.
      if (params.requireLastTradeBuy && lastTradeBuy.get(event.mint) !== true) continue;
      // No ceiling on how far it ran: momentum persists, the biggest
      // early pumpers peak highest.
      if (now > (note.firstPrice * params.maxEntryMultipleBps) / 10_000) continue;
      const cap = state.maxBuyLamports(event.mint);
      const amount = cap < params.stakeLamports ? cap : params.stakeLamports;
      if (amount < MIN_ROUND_TRIP_LAMPORTS) continue;
      if (orders.length >= state.hello.caps.max_orders_per_batch) continue;
      if (slots <= 0 || free < amount + tx) continue;
      orders.push({
        side: 'buy',
        mint: event.mint,
        quote_lamports: amount,
        max_slippage_bps: params.buySlippageBps,
      });
      free -= amount + tx;
      slots -= 1;
    }

    return orders;
  }

  return { onBatch };
}

// Run as the grader runs it: `node /bot/main.ts`. Guarded so the strategy
// can also be imported (for local backtests) without starting the wire loop.
const bot = createBot(DEFAULT_PARAMS);
const invokedAsMain =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedAsMain) {
  await runBot({ onBatch: bot.onBatch });
}
