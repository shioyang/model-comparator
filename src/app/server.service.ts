import { Injectable } from '@angular/core';
import { Headers, RequestOptions, Http, URLSearchParams } from '@angular/http';
import { Observable } from 'rxjs/Observable';
import 'rxjs/Rx'; // for operators


@Injectable()
export class ServerService {
  private baseUrl = 'users';

  constructor(private http: Http) { }

  predictImage(image_path: string): Observable<any> {
    let searchParams = new URLSearchParams();
    searchParams.set('file_path', './' + image_path);
    return this.http.get(this.baseUrl + '/predict', { search: searchParams })
        .map(res => {
            return res
            // return res.json()
        })
  }

}