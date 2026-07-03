import { redirect } from "next/navigation";
import { AdminPanel } from "./admin-panel";
import { sessionIsAdmin } from "@/lib/admin-role";
import { getSessionFromCookies } from "@/lib/session";

export default async function AdminPage() {
  const session = await getSessionFromCookies();
  if (!session) redirect("/login");
  if (!sessionIsAdmin(session)) redirect("/");
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <AdminPanel />
    </div>
  );
}
