ALTER TABLE `episodes` ADD `thumb_key` text;--> statement-breakpoint
ALTER TABLE `episodes` ADD `duration` integer;--> statement-breakpoint
ALTER TABLE `episodes` ADD `views` integer DEFAULT 0 NOT NULL;--> statement-breakpoint
ALTER TABLE `uploads` ADD `thumb_key` text;--> statement-breakpoint
ALTER TABLE `uploads` ADD `duration` integer;