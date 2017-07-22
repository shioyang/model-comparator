import { BrowserModule } from '@angular/platform-browser'
import { NgModule } from '@angular/core'
import { HttpModule } from '@angular/http';
import { FormsModule } from '@angular/forms'
import { BrowserAnimationsModule } from '@angular/platform-browser/animations'
import { MdSelectModule, MdOptionModule, MdButtonModule } from '@angular/material'

import { AppComponent } from './app.component'
import { ServerService } from './server.service'

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    HttpModule,
    FormsModule,
    BrowserAnimationsModule,   // Animation module for Angular Material
    MdSelectModule,            // md-select
    MdOptionModule,            // md-option
    MdButtonModule
  ],
  providers: [ServerService],
  bootstrap: [AppComponent]
})
export class AppModule { }
