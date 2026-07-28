ALTER TABLE "Repository" DROP COLUMN IF EXISTS "indexProgress";

ALTER TABLE "RepositoryIndexJob" DROP COLUMN IF EXISTS "progress";

UPDATE "RepositoryIndexJob"
SET "trigger" = 'RETRY'
WHERE "trigger" = 'PUSH';

ALTER TYPE "RepositoryIndexJobTrigger" RENAME TO "RepositoryIndexJobTrigger_old";
CREATE TYPE "RepositoryIndexJobTrigger" AS ENUM ('CONNECT', 'RETRY');

ALTER TABLE "RepositoryIndexJob"
  ALTER COLUMN "trigger" TYPE "RepositoryIndexJobTrigger"
  USING "trigger"::text::"RepositoryIndexJobTrigger";

DROP TYPE "RepositoryIndexJobTrigger_old";
