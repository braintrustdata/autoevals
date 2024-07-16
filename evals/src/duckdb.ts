import type { TableData, Connection } from "duckdb";
import * as duckdb from "duckdb";

let _duckdb_db: duckdb.Database | null = null;
export function getDuckDBConn() {
  if (!_duckdb_db) {
    _duckdb_db = new duckdb.Database(":memory:");
  }
  return _duckdb_db.connect();
}

export async function duckq(con: Connection, sql: string): Promise<TableData> {
  return new Promise((resolve, reject) => {
    con.all(sql, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows);
      }
    });
  });
}
