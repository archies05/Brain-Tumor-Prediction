import { Client, Databases, Account, ID } from "appwrite";

const client = new Client();
client
  .setEndpoint("https://cloud.appwrite.io/v1")
  .setProject("678be84e00185d560281");

export const account = new Account(client);
export const databases = new Databases(client);
