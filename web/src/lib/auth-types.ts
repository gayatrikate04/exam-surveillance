export type StoredUser = {
  id: string;
  email: string;
  passwordHash: string;
  salt: string;
  createdAt: string;
  role?: "admin" | "user";
};
