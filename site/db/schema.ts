import { sql } from "drizzle-orm";
import { integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

export const series = sqliteTable("series", {
  id: text("id").primaryKey(),
  title: text("title").notNull(),
  synopsis: text("synopsis").notNull().default(""),
  genre: text("genre").notNull().default("Anime"),
  coverKey: text("cover_key"),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});

export const episodes = sqliteTable("episodes", {
  id: text("id").primaryKey(),
  seriesId: text("series_id").notNull().references(() => series.id),
  number: integer("number").notNull(),
  title: text("title").notNull(),
  objectKey: text("object_key").notNull(),
  contentType: text("content_type").notNull(),
  bytes: integer("bytes").notNull(),
  thumbKey: text("thumb_key"),
  duration: integer("duration"),
  views: integer("views").notNull().default(0),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
}, (table) => [uniqueIndex("episode_series_number_unique").on(table.seriesId, table.number)]);

export const uploads = sqliteTable("uploads", {
  id: text("id").primaryKey(),
  ownerEmail: text("owner_email").notNull(),
  seriesId: text("series_id").notNull().references(() => series.id),
  number: integer("number").notNull(),
  title: text("title").notNull(),
  objectKey: text("object_key").notNull(),
  uploadId: text("upload_id").notNull(),
  contentType: text("content_type").notNull(),
  bytes: integer("bytes").notNull(),
  thumbKey: text("thumb_key"),
  duration: integer("duration"),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});
