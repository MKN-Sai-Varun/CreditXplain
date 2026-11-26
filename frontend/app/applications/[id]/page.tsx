import { Suspense } from "react";

import { ApplicationView } from "@/components/application/ApplicationView";
import { Spinner } from "@/components/ui";

// The id is only known at request time, so the client view that reads it
// sits inside a Suspense boundary (required with cacheComponents).
export default function ApplicationPage() {
  return (
    <Suspense fallback={<Spinner />}>
      <ApplicationView />
    </Suspense>
  );
}
