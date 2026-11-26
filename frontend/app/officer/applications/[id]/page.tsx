import { Suspense } from "react";

import { OfficerReview } from "@/components/officer/OfficerReview";
import { Spinner } from "@/components/ui";

// The id is only known at request time, so the client view that reads it
// sits inside a Suspense boundary (required with cacheComponents).
export default function OfficerApplicationPage() {
  return (
    <Suspense fallback={<Spinner />}>
      <OfficerReview />
    </Suspense>
  );
}
